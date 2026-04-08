#!/usr/bin/env python3
"""
Надёжный пайплайн Manuscript OCR с:
- адаптивным batch для TRBA,
- fallback на CPU при OOM,
- кешированием детекции/распознавания,
- логированием GPU памяти,
- сохранением кропов слов для self-supervised.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch

try:
    import xxhash

    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False

try:
    import pynvml

    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

from manuscript.detectors import EAST
from manuscript.recognizers import TRBA
from manuscript.data import Page
from manuscript.utils import read_image

logger = logging.getLogger("safe_manuscript")


# ---------- Вспомогательные структуры ----------

@dataclass
class PageStats:
    page_id: str
    image_path: str
    image_name: str
    width: int
    height: int
    num_words: int
    avg_word_area: float
    max_word_area: float


@dataclass
class MemoryStats:
    total_mb: int
    used_mb: int
    free_mb: int
    torch_alloc_mb: int
    other_mb: int


# ---------- Хеширование ----------

def compute_page_id(image_path: Path) -> str:
    """Быстрый хеш от пути, размера и mtime. xxh128 если доступен, иначе md5."""
    stat = image_path.stat()
    key = f"{image_path}|{stat.st_size}|{int(stat.st_mtime)}"
    if XXHASH_AVAILABLE:
        return xxhash.xxh128_hexdigest(key.encode("utf-8"))[:16]
    import hashlib
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:16]


# ---------- Утилиты для GPU памяти ----------

def get_gpu_memory(device_index: int = 0) -> Optional[MemoryStats]:
    if not torch.cuda.is_available():
        return None

    try:
        if NVML_AVAILABLE:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total = mem.total
            used = mem.used
            free = mem.free
        else:
            props = torch.cuda.get_device_properties(device_index)
            total = props.total_memory
            reserved = torch.cuda.memory_reserved(device_index)
            used = reserved
            free = total - reserved

        torch_alloc = torch.cuda.memory_allocated(device_index)
        other = max(0, used - torch_alloc)

        mb = lambda x: int(x / (1024 * 1024))
        return MemoryStats(
            total_mb=mb(total),
            used_mb=mb(used),
            free_mb=mb(free),
            torch_alloc_mb=mb(torch_alloc),
            other_mb=mb(other),
        )
    except Exception as e:
        logger.warning(f"Не удалось прочитать GPU память: {e}")
        return None


# ---------- OOM-безопасный TRBA ----------

def trba_predict_safe(
    trba: TRBA,
    images: List[np.ndarray],
    init_batch: int = 32,
    min_batch: int = 1,
    allow_fallback_cpu: bool = True,
    gpu_index: int = 0,
    log_oom_cb=None,
    page_stats: Optional[PageStats] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Безопасный вызов TRBA.predict с уменьшением batch и fallback на CPU.
    """
    results: List[Dict[str, Any]] = []
    remaining = list(images)
    current_batch = max(1, min(init_batch, len(remaining)))
    orig_is_cuda = getattr(trba, "device", None) == "cuda"
    used_cuda = orig_is_cuda

    while remaining:
        batch_imgs = remaining[:current_batch]
        try:
            batch_res = trba.predict(batch_imgs, batch_size=current_batch)
            results.extend(batch_res)
            remaining = remaining[current_batch:]
        except Exception as e:
            msg = str(e).lower()
            is_oom = (
                "out of memory" in msg
                or "cuda failure 2" in msg
                or "cudaerrormemoryallocation" in msg
            )

            if not is_oom:
                raise

            mem_stats = get_gpu_memory(gpu_index)
            if log_oom_cb is not None:
                log_oom_cb(
                    stage="trba",
                    oom=True,
                    batch_size=current_batch,
                    mem_stats=mem_stats,
                    page_stats=page_stats,
                    used_cuda=used_cuda,
                )

            logger.warning(
                f"OOM в TRBA (batch={current_batch}). "
                f"Сообщение: {e}"
            )

            if current_batch > min_batch:
                current_batch = max(min_batch, current_batch // 2)
                logger.info(f"Пробуем ещё раз с меньшим batch={current_batch}")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue

            if allow_fallback_cpu and used_cuda:
                logger.warning("Переходим на TRBA(device='cpu') после OOM.")
                trba = TRBA(
                    weights=trba.weights,
                    config=getattr(trba, "config_path", None),
                    charset=getattr(trba, "charset_path", None),
                    device="cpu",
                )
                used_cuda = False
                continue

            raise

    return results, {"final_batch_size": current_batch, "used_cuda": used_cuda}


# ---------- Word-кропы и статистика страницы ----------

def extract_word_crops(
    page: Page,
    image_array: np.ndarray,
    min_text_size: int = 5,
    rotate_threshold: float = 1.5,
) -> Tuple[List[np.ndarray], List[Any], PageStats]:
    word_images: List[np.ndarray] = []
    word_objects: List[Any] = []
    areas: List[int] = []

    h_img, w_img = image_array.shape[:2]

    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                poly = np.array(word.polygon, dtype=np.int32)
                x_min, y_min = np.min(poly, axis=0)
                x_max, y_max = np.max(poly, axis=0)
                width = x_max - x_min
                height = y_max - y_min

                if width < min_text_size or height < min_text_size:
                    continue

                x1 = max(0, int(x_min))
                y1 = max(0, int(y_min))
                x2 = min(w_img, int(x_max))
                y2 = min(h_img, int(y_max))
                crop = image_array[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                if rotate_threshold and crop.shape[0] > crop.shape[1] * rotate_threshold:
                    crop = np.rot90(crop, k=-1)

                word_images.append(crop)
                word_objects.append(word)
                areas.append(int(width * height))

    num_words = len(word_images)
    avg_area = float(np.mean(areas)) if areas else 0.0
    max_area = float(np.max(areas)) if areas else 0.0

    page_stats = PageStats(
        page_id="",
        image_path="",
        image_name="",
        width=w_img,
        height=h_img,
        num_words=num_words,
        avg_word_area=avg_area,
        max_word_area=max_area,
    )
    return word_images, word_objects, page_stats


# ---------- Self-supervised: сохранение кропов ----------

def sanitize_text_for_path(text: str) -> str:
    bad = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    t = text.strip().lower().replace(" ", "_")
    for b in bad:
        t = t.replace(b, "")
    return t or "unk"


def save_word_for_selftrain(
    crop: np.ndarray,
    text: str,
    conf: float,
    doc_id: str,
    page_index: int,
    word_index: int,
    out_dir: Path,
    bucket: str = "high_conf",
) -> Optional[Path]:
    out_dir = Path(out_dir)
    cls = sanitize_text_for_path(text)
    first = cls[0] if cls else "_"

    subdir = out_dir / bucket / first / cls
    subdir.mkdir(parents=True, exist_ok=True)

    fname = f"conf{conf:.3f}_{doc_id}_p{page_index:04d}_w{word_index:05d}.png"
    fpath = subdir / fname

    img = Image.fromarray(crop)
    img.save(fpath)
    return fpath


# ---------- Основной конвейер для одной страницы ----------

def process_image(
    image_path: Path,
    detector: EAST,
    recognizer: TRBA,
    cache_dir: Path,
    log_dir: Path,
    selftrain_dir: Optional[Path],
    init_batch: int = 32,
    min_batch: int = 4,
    gpu_index: int = 0,
    conf_high: float = 0.97,
    conf_mid: float = 0.9,
    min_text_size: int = 5,
    rotate_threshold: float = 1.5,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if selftrain_dir is not None:
        selftrain_dir.mkdir(parents=True, exist_ok=True)

    image_name = image_path.name
    page_id = compute_page_id(image_path)

    det_cache = cache_dir / f"{page_id}_det.json"
    full_cache = cache_dir / f"{page_id}_full.json"
    txt_out = cache_dir / f"{page_id}.txt"

    if full_cache.exists() and txt_out.exists():
        logger.info(f"[{image_name}] уже обработан, пропускаем.")
        return

    # --- Загрузка детекции (или запуск EAST) ---
    if det_cache.exists():
        logger.info(f"[{image_name}] читаем детекцию из кеша.")
        with open(det_cache, "r", encoding="utf-8") as f:
            page_data = json.load(f)
        page = Page.model_validate(page_data)
    else:
        logger.info(f"[{image_name}] запускаем EAST.")
        mem_before = get_gpu_memory(gpu_index)
        det_start = time.time()
        det_result = detector.predict(
            str(image_path),
            return_maps=False,
            sort_reading_order=True,
            split_into_columns=True,
        )
        det_time = time.time() - det_start
        page: Page = det_result["page"]
        page_dict = page.model_dump(mode="json")
        page_dict["_image_name"] = image_name
        with open(det_cache, "w", encoding="utf-8") as f:
            json.dump(page_dict, f, ensure_ascii=False)
        mem_after = get_gpu_memory(gpu_index)
        logger.info(
            f"[{image_name}] EAST: {det_time:.3f}s, "
            f"GPU до/после: {mem_before} -> {mem_after}"
        )

    # --- Кропы слов ---
    img_array = read_image(str(image_path))
    word_images, word_objects, page_stats = extract_word_crops(
        page, img_array, min_text_size=min_text_size, rotate_threshold=rotate_threshold
    )
    page_stats.page_id = page_id
    page_stats.image_path = str(image_path)
    page_stats.image_name = image_name

    if not word_images:
        logger.info(f"[{image_name}] нет слов для распознавания.")
        page_dict = page.model_dump(mode="json")
        page_dict["_image_name"] = image_name
        with open(full_cache, "w", encoding="utf-8") as f:
            json.dump(page_dict, f, ensure_ascii=False)
        with open(txt_out, "w", encoding="utf-8") as f:
            f.write("")
        return

    # --- Логирование OOM / памяти в CSV ---
    oom_log_path = log_dir / "oom_stats.csv"
    new_file = not oom_log_path.exists()
    oom_csv = open(oom_log_path, "a", newline="", encoding="utf-8")
    oom_writer = csv.writer(oom_csv)
    if new_file:
        oom_writer.writerow(
            [
                "timestamp",
                "page_id",
                "image_name",
                "image_path",
                "stage",
                "oom",
                "fallback_to_cpu",
                "init_batch",
                "final_batch",
                "width",
                "height",
                "num_words",
                "avg_word_area",
                "max_word_area",
                "gpu_total_mb",
                "gpu_used_mb",
                "gpu_free_mb",
                "torch_alloc_mb",
                "other_mb",
            ]
        )

    def log_oom_cb(stage, oom, batch_size, mem_stats, page_stats, used_cuda):
        oom_writer.writerow(
            [
                time.time(),
                page_stats.page_id,
                page_stats.image_name,
                page_stats.image_path,
                stage,
                int(oom),
                int(not used_cuda),
                batch_size,
                batch_size,
                page_stats.width,
                page_stats.height,
                page_stats.num_words,
                page_stats.avg_word_area,
                page_stats.max_word_area,
                mem_stats.total_mb if mem_stats else "",
                mem_stats.used_mb if mem_stats else "",
                mem_stats.free_mb if mem_stats else "",
                mem_stats.torch_alloc_mb if mem_stats else "",
                mem_stats.other_mb if mem_stats else "",
            ]
        )
        oom_csv.flush()

    # --- TRBA с OOM-защитой ---
    logger.info(
        f"[{image_name}] TRBA: {len(word_images)} слов, init_batch={init_batch}."
    )
    trba_start = time.time()
    rec_results, rec_info = trba_predict_safe(
        recognizer,
        word_images,
        init_batch=init_batch,
        min_batch=min_batch,
        allow_fallback_cpu=True,
        gpu_index=gpu_index,
        log_oom_cb=log_oom_cb,
        page_stats=page_stats,
    )
    trba_time = time.time() - trba_start
    logger.info(
        f"[{image_name}] TRBA: {trba_time:.3f}s, "
        f"final_batch={rec_info['final_batch_size']}, "
        f"used_cuda={rec_info['used_cuda']}"
    )

    # Присваиваем текст и confidence словам, сохраняем кропы для self-train
    avg_conf = []
    for idx, (word_obj, rec, crop) in enumerate(
        zip(word_objects, rec_results, word_images)
    ):
        text = rec["text"]
        conf = float(rec["confidence"])
        word_obj.text = text
        word_obj.recognition_confidence = conf
        avg_conf.append(conf)

        if selftrain_dir is not None and text:
            if conf >= conf_high:
                bucket = "high_conf"
            elif conf >= conf_mid:
                bucket = "mid_conf"
            else:
                bucket = None
            if bucket is not None:
                save_word_for_selftrain(
                    crop,
                    text,
                    conf,
                    doc_id=page_id,
                    page_index=0,
                    word_index=idx,
                    out_dir=selftrain_dir,
                    bucket=bucket,
                )

    # --- Сохраняем страницу и текст ---
    page_dict = page.model_dump(mode="json")
    page_dict["_image_name"] = image_name
    with open(full_cache, "w", encoding="utf-8") as f:
        json.dump(page_dict, f, ensure_ascii=False)

    lines = []
    for block in page.blocks:
        for line in block.lines:
            texts = [w.text for w in line.words if w.text]
            if texts:
                lines.append(" ".join(texts))
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # финальная запись строки статистики без OOM
    mem_stats = get_gpu_memory(gpu_index)
    oom_writer.writerow(
        [
            time.time(),
            page_stats.page_id,
            page_stats.image_name,
            page_stats.image_path,
            "trba_done",
            0,
            int(not rec_info["used_cuda"]),
            init_batch,
            rec_info["final_batch_size"],
            page_stats.width,
            page_stats.height,
            page_stats.num_words,
            page_stats.avg_word_area,
            page_stats.max_word_area,
            mem_stats.total_mb if mem_stats else "",
            mem_stats.used_mb if mem_stats else "",
            mem_stats.free_mb if mem_stats else "",
            mem_stats.torch_alloc_mb if mem_stats else "",
            mem_stats.other_mb if mem_stats else "",
        ]
    )
    oom_csv.close()


# ---------- CLI ----------

def iter_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".jp2"}
    return [
        p
        for p in sorted(input_path.rglob("*"))
        if p.is_file() and p.suffix.lower() in exts
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Файл или директория с изображениями")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Основное устройство для моделей (cuda/cpu)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="cache", help="Каталог для JSON/текста"
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs", help="Каталог для CSV‑логов"
    )
    parser.add_argument(
        "--selftrain-dir",
        type=str,
        default=None,
        help="Каталог для self-supervised кропов (если не указан — не сохраняем)",
    )
    parser.add_argument(
        "--init-batch",
        type=int,
        default=32,
        help="Начальный размер batch для TRBA",
    )
    parser.add_argument(
        "--min-batch",
        type=int,
        default=4,
        help="Минимальный размер batch для TRBA перед fallback на CPU",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="Индекс GPU для логирования памяти",
    )
    parser.add_argument(
        "--conf-high",
        type=float,
        default=0.97,
        help="Порог confidence для high_conf self-train",
    )
    parser.add_argument(
        "--conf-mid",
        type=float,
        default=0.90,
        help="Порог для mid_conf self-train",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    input_path = Path(args.input)
    cache_dir = Path(args.cache_dir)
    log_dir = Path(args.log_dir)
    selftrain_dir = Path(args.selftrain_dir) if args.selftrain_dir else None

    # Инициализация моделей
    logger.info(f"Инициализируем EAST(device={args.device}) и TRBA(device={args.device})")
    detector = EAST(device=args.device)
    recognizer = TRBA(weights="trba_base_g1", device=args.device)

    images = iter_images(input_path)
    logger.info(f"Найдено {len(images)} изображений.")

    for img in images:
        try:
            process_image(
                img,
                detector,
                recognizer,
                cache_dir=cache_dir,
                log_dir=log_dir,
                selftrain_dir=selftrain_dir,
                init_batch=args.init_batch,
                min_batch=args.min_batch,
                gpu_index=args.gpu_index,
                conf_high=args.conf_high,
                conf_mid=args.conf_mid,
            )
        except Exception as e:
            logger.exception(f"Ошибка при обработке {img.name}: {e}")


if __name__ == "__main__":
    main()
