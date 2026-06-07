#!/usr/bin/env python3
"""
Adaptive batch OCR with CuPy, no CPU fallback unless OOM at batch=1.
"""

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cupy as cp
import torch
from PIL import Image

# ----- Dynamic backend selection -----
def import_backend(backend: str):
    if backend == "cupy":
        import cupy as xp
        cupy_available = True
        def asnumpy(arr):
            return xp.asnumpy(arr)
    else:
        import numpy as xp
        cupy_available = False
        def asnumpy(arr):
            return arr
    return xp, cupy_available, asnumpy

xp = None
cupy_available = False
asnumpy = None

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


# ---------- Adaptive Batch Sizer (no hard cap, learns safe max) ----------
class AdaptiveBatchSizer:
    def __init__(self, init_batch: int, min_batch: int = 1,
                 increase_factor: float = 1.2, decrease_factor: float = 0.7,
                 stable_rounds: int = 2):
        self.current = init_batch
        self.min_batch = min_batch
        self.inc_factor = increase_factor
        self.dec_factor = decrease_factor
        self.stable_rounds = stable_rounds
        self.success_streak = 0
        self.oom_count = 0
        self.optimal = init_batch
        self.global_safe_max = init_batch * 4   # generous initial

    def on_success(self, actual_batch_used: int):
        if self.current >= self.global_safe_max:
            return
        self.success_streak += 1
        if self.success_streak >= self.stable_rounds:
            new_batch = min(self.global_safe_max, int(self.current * self.inc_factor))
            if new_batch > self.current:
                logger.info(f"📈 Increasing batch size: {self.current} → {new_batch}")
                self.current = new_batch
                self.success_streak = 0
        self.optimal = max(self.optimal, actual_batch_used)

    def on_oom(self, batch_that_failed: int):
        self.oom_count += 1
        self.success_streak = 0
        self.global_safe_max = max(self.min_batch, int(batch_that_failed * self.dec_factor))
        new_batch = min(self.global_safe_max, self.current)
        if new_batch != self.current:
            logger.warning(f"💥 OOM at batch {batch_that_failed}. New safe max = {self.global_safe_max}, reducing to {new_batch}")
            self.current = new_batch
        self.success_streak = 0

    def get_current(self) -> int:
        return self.current


# ---------- GPU Memory Monitor ----------
class GPUMemoryMonitor:
    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.peak_used_mb = 0

    def get_stats(self):
        if not cupy_available or not torch.cuda.is_available():
            return None
        try:
            if NVML_AVAILABLE:
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total = mem.total
                used = mem.used
                free = mem.free
            else:
                props = torch.cuda.get_device_properties(self.device_index)
                total = props.total_memory
                reserved = torch.cuda.memory_reserved(self.device_index)
                used = reserved
                free = total - reserved
            torch_alloc = torch.cuda.memory_allocated(self.device_index)
            other = max(0, used - torch_alloc)
            mb = lambda x: int(x / (1024 * 1024))
            used_mb = mb(used)
            if used_mb > self.peak_used_mb:
                self.peak_used_mb = used_mb
            return type('Stats', (), {
                'total_mb': mb(total),
                'used_mb': used_mb,
                'free_mb': mb(free),
                'torch_alloc_mb': mb(torch_alloc),
                'other_mb': mb(other),
                'peak_used_mb': self.peak_used_mb
            })()
        except Exception as e:
            logger.warning(f"Failed to read GPU memory: {e}")
            return None


# ---------- Safe TRBA predict (no CPU fallback unless batch=1) ----------
def trba_predict_safe(
    trba: TRBA,
    images: List,
    batch_sizer: AdaptiveBatchSizer,
    allow_fallback_cpu: bool = True,
    gpu_index: int = 0,
    log_oom_cb=None,
    page_stats=None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    results = []
    remaining_indices = list(range(len(images)))
    current_batch = batch_sizer.get_current()
    oom_occurred = False
    used_cuda = getattr(trba, "device", None) == "cuda"
    batch_reductions = 0
    final_batch = current_batch
    total_predict_time = 0.0

    i = 0
    while i < len(remaining_indices):
        batch_end = min(i + current_batch, len(remaining_indices))
        batch_idxs = remaining_indices[i:batch_end]
        batch_crops = [images[idx] for idx in batch_idxs]

        # Convert CuPy -> NumPy
        np_batch = []
        for crop in batch_crops:
            if cupy_available and isinstance(crop, xp.ndarray):
                np_batch.append(cp.asnumpy(crop))
            else:
                np_batch.append(crop)

        # Retry loop for OOM (especially at batch=1)
        retry_count = 0
        max_retries = 3
        while True:
            try:
                start = time.perf_counter()
                batch_res = trba.predict(np_batch, batch_size=len(np_batch))
                batch_time = time.perf_counter() - start
                total_predict_time += batch_time
                results.extend(batch_res)
                i = batch_end
                batch_sizer.on_success(current_batch)
                final_batch = current_batch
                break   # success, exit retry loop
            except Exception as e:
                msg = str(e).lower()
                is_oom = ("out of memory" in msg or "cuda failure 2" in msg or
                          "cudaerrormemoryallocation" in msg or "failed to allocate memory" in msg)
                if not is_oom:
                    raise
                oom_occurred = True
                batch_reductions += 1

                # Log OOM
                mem_stats = None
                if cupy_available:
                    monitor = GPUMemoryMonitor(gpu_index)
                    mem_stats = monitor.get_stats()
                if log_oom_cb:
                    log_oom_cb("trba", True, current_batch, mem_stats, page_stats, used_cuda)

                logger.warning(f"OOM at batch {current_batch}. Reducing.")

                # If batch size is 1, try to clear memory and retry a few times
                if current_batch == 1:
                    retry_count += 1
                    if retry_count <= max_retries:
                        logger.warning(f"OOM at batch=1, retry {retry_count}/{max_retries} after clearing cache.")
                        if cupy_available:
                            torch.cuda.empty_cache()
                            cp._default_memory_pool.free_all_blocks()
                            cp.cuda.Device(gpu_index).synchronize()
                        time.sleep(1)   # give GPU time to recover
                        continue        # retry the same batch
                    elif allow_fallback_cpu and used_cuda:
                        logger.warning(f"Still OOM at batch=1 after {max_retries} retries. Falling back to CPU.")
                        trba = TRBA(
                            weights=trba.weights,
                            config=getattr(trba, "config_path", None),
                            charset=getattr(trba, "charset_path", None),
                            device="cpu",
                        )
                        used_cuda = False
                        continue        # retry this batch with CPU
                    else:
                        raise RuntimeError("OOM at batch=1 and CPU fallback disabled/unsuccessful after retries.")

                # For batch size > 1, reduce the batch size and retry
                batch_sizer.on_oom(current_batch)
                current_batch = batch_sizer.get_current()
                if current_batch < 1:
                    raise RuntimeError("Batch size below minimum.")
                if cupy_available:
                    torch.cuda.empty_cache()
                # break out of inner retry loop; outer loop will retry with smaller batch
                break

    return results, {
        "final_batch_size": final_batch,
        "used_cuda": used_cuda,
        "batch_reductions": batch_reductions,
        "oom_occurred": oom_occurred,
        "optimal_batch": batch_sizer.optimal,
        "predict_time_ms": total_predict_time * 1000,
    }

# ---------- Word crop extraction (unchanged) ----------
def extract_word_crops(
    page: Page,
    image_array,
    min_text_size: int = 5,
    rotate_threshold: float = 1.5,
):
    word_images = []
    word_objects = []
    areas = []
    h_img, w_img = image_array.shape[:2]
    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                poly = xp.array(word.polygon, dtype=xp.int32)
                x_min, y_min = xp.min(poly, axis=0)
                x_max, y_max = xp.max(poly, axis=0)
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
                    crop = xp.rot90(crop, k=-1)
                word_images.append(crop)
                word_objects.append(word)
                areas.append(int(width * height))
    num_words = len(word_images)
    if areas:
        areas_arr = xp.array(areas)
        avg_area = float(xp.mean(areas_arr))
        max_area = float(xp.max(areas_arr))
        min_area = float(xp.min(areas_arr))
    else:
        avg_area = max_area = min_area = 0.0
    page_stats = PageStats(
        page_id="", image_path="", image_name="",
        width=w_img, height=h_img, num_words=num_words,
        avg_word_area=avg_area, max_word_area=max_area, min_word_area=min_area,
    )
    return word_images, word_objects, page_stats


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
    min_word_area: float = 0.0


@dataclass
class PerformanceRecord:
    timestamp: float
    variant: str
    page_id: str
    image_name: str
    image_path: str
    width: int
    height: int
    num_words: int
    avg_word_area: float
    max_word_area: float
    min_word_area: float
    time_detection: float
    time_cropping: float
    time_recognition: float
    time_total: float
    throughput_detection: float
    throughput_recognition: float
    throughput_total: float
    gpu_mem_before_detection: Optional[int]
    gpu_mem_after_detection: Optional[int]
    gpu_mem_before_recognition: Optional[int]
    gpu_mem_after_recognition: Optional[int]
    gpu_mem_peak: Optional[int]
    init_batch: int
    final_batch: int
    batch_reductions: int
    fallback_to_cpu: bool
    oom_occurred: bool
    used_cuda_final: bool
    optimal_batch: int = 0
    convert_time_ms: float = 0.0
    predict_time_ms: float = 0.0


def compute_page_id(image_path: Path) -> str:
    stat = image_path.stat()
    key = f"{image_path}|{stat.st_size}|{int(stat.st_mtime)}"
    if XXHASH_AVAILABLE:
        return xxhash.xxh128_hexdigest(key.encode("utf-8"))[:16]
    import hashlib
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:16]


def sanitize_text_for_path(text: str) -> str:
    bad = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    t = text.strip().lower().replace(" ", "_")
    for b in bad:
        t = t.replace(b, "")
    return t or "unk"


def save_word_for_selftrain(crop, text, conf, doc_id, page_index, word_index, out_dir, bucket="high_conf"):
    out_dir = Path(out_dir)
    cls = sanitize_text_for_path(text)
    first = cls[0] if cls else "_"
    subdir = out_dir / bucket / first / cls
    subdir.mkdir(parents=True, exist_ok=True)
    fname = f"conf{conf:.3f}_{doc_id}_p{page_index:04d}_w{word_index:05d}.png"
    fpath = subdir / fname
    img_np = asnumpy(crop)
    img = Image.fromarray(img_np)
    img.save(fpath)
    return fpath


def process_image(
    image_path: Path,
    detector: EAST,
    recognizer: TRBA,
    batch_sizer: AdaptiveBatchSizer,
    cache_dir: Path,
    log_dir: Path,
    selftrain_dir: Optional[Path],
    variant: str,
    init_batch: int,
    min_batch: int,
    gpu_index: int,
    conf_high: float,
    conf_mid: float,
    min_text_size: int,
    rotate_threshold: float,
    record_performance: bool = True,
) -> Optional[PerformanceRecord]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if selftrain_dir:
        selftrain_dir.mkdir(parents=True, exist_ok=True)

    image_name = image_path.name
    page_id = compute_page_id(image_path)

    det_cache = cache_dir / f"{page_id}_det.json"
    full_cache = cache_dir / f"{page_id}_full.json"
    txt_out = cache_dir / f"{page_id}.txt"

    if full_cache.exists() and txt_out.exists():
        logger.info(f"[{image_name}] already processed, skipping.")
        return None

    mem_monitor = GPUMemoryMonitor(gpu_index) if cupy_available else None
    perf = PerformanceRecord(
        timestamp=time.time(), variant=variant, page_id=page_id, image_name=image_name, image_path=str(image_path),
        width=0, height=0, num_words=0, avg_word_area=0.0, max_word_area=0.0, min_word_area=0.0,
        time_detection=0.0, time_cropping=0.0, time_recognition=0.0, time_total=0.0,
        throughput_detection=0.0, throughput_recognition=0.0, throughput_total=0.0,
        gpu_mem_before_detection=None, gpu_mem_after_detection=None,
        gpu_mem_before_recognition=None, gpu_mem_after_recognition=None, gpu_mem_peak=None,
        init_batch=init_batch, final_batch=init_batch, batch_reductions=0,
        fallback_to_cpu=False, oom_occurred=False, used_cuda_final=cupy_available and torch.cuda.is_available(),
        optimal_batch=0, convert_time_ms=0.0, predict_time_ms=0.0
    )

    start_total = time.time()

    # --- Detection ---
    if det_cache.exists():
        logger.info(f"[{image_name}] loading detection from cache.")
        with open(det_cache, "r", encoding="utf-8") as f:
            page_data = json.load(f)
        page = Page.model_validate(page_data)
        perf.time_detection = 0.0
    else:
        logger.info(f"[{image_name}] running EAST.")
        if mem_monitor:
            mb = mem_monitor.get_stats()
            perf.gpu_mem_before_detection = mb.used_mb if mb else None
        det_start = time.time()
        det_result = detector.predict(
            str(image_path), return_maps=False,
            sort_reading_order=True, split_into_columns=True,
        )
        perf.time_detection = time.time() - det_start
        page = det_result["page"]
        page_dict = page.model_dump(mode="json")
        page_dict["_image_name"] = image_name
        with open(det_cache, "w", encoding="utf-8") as f:
            json.dump(page_dict, f, ensure_ascii=False)
        if mem_monitor:
            ma = mem_monitor.get_stats()
            perf.gpu_mem_after_detection = ma.used_mb if ma else None
        logger.info(f"[{image_name}] EAST: {perf.time_detection:.3f}s")

    # --- Cropping ---
    crop_start = time.time()
    img_np = read_image(str(image_path))
    img_array = xp.asarray(img_np)
    word_images, word_objects, page_stats = extract_word_crops(
        page, img_array, min_text_size=min_text_size, rotate_threshold=rotate_threshold
    )
    perf.time_cropping = time.time() - crop_start
    perf.width = page_stats.width
    perf.height = page_stats.height
    perf.num_words = page_stats.num_words
    perf.avg_word_area = page_stats.avg_word_area
    perf.max_word_area = page_stats.max_word_area
    perf.min_word_area = page_stats.min_word_area
    page_stats.page_id = page_id
    page_stats.image_path = str(image_path)
    page_stats.image_name = image_name

    if not word_images:
        logger.info(f"[{image_name}] no words to recognise.")
        page_dict = page.model_dump(mode="json")
        page_dict["_image_name"] = image_name
        with open(full_cache, "w", encoding="utf-8") as f:
            json.dump(page_dict, f, ensure_ascii=False)
        with open(txt_out, "w", encoding="utf-8") as f:
            f.write("")
        perf.time_total = time.time() - start_total
        if record_performance:
            log_performance(perf, log_dir)
        return perf

    # --- OOM logging setup ---
    oom_log_path = log_dir / "oom_stats.csv"
    new_file = not oom_log_path.exists()
    oom_csv = open(oom_log_path, "a", newline="", encoding="utf-8")
    oom_writer = csv.writer(oom_csv)
    if new_file:
        oom_writer.writerow([
            "timestamp", "variant", "page_id", "image_name", "image_path",
            "stage", "oom", "fallback_to_cpu", "init_batch", "final_batch",
            "width", "height", "num_words", "avg_word_area", "max_word_area", "min_word_area",
            "gpu_total_mb", "gpu_used_mb", "gpu_free_mb", "torch_alloc_mb", "other_mb", "gpu_peak_mb"
        ])

    def log_oom_cb(stage, oom, batch_size, mem_stats, page_stats, used_cuda):
        total_mb = used_mb = free_mb = torch_alloc_mb = other_mb = peak_mb = ""
        if mem_stats is not None:
            if hasattr(mem_stats, 'total_mb'):
                total_mb = mem_stats.total_mb
                used_mb = mem_stats.used_mb
                free_mb = mem_stats.free_mb
                torch_alloc_mb = mem_stats.torch_alloc_mb
                other_mb = mem_stats.other_mb
                peak_mb = mem_stats.peak_used_mb
        oom_writer.writerow([
            time.time(), variant, page_stats.page_id, page_stats.image_name, page_stats.image_path,
            stage, int(oom), int(not used_cuda), init_batch, batch_size,
            page_stats.width, page_stats.height, page_stats.num_words,
            page_stats.avg_word_area, page_stats.max_word_area, page_stats.min_word_area,
            total_mb, used_mb, free_mb, torch_alloc_mb, other_mb, peak_mb
        ])
        oom_csv.flush()

    # --- TRBA with adaptive batch (NO CPU FALLBACK unless batch=1) ---
    logger.info(f"[{image_name}] TRBA: {len(word_images)} words, current batch={batch_sizer.get_current()}.")
    if mem_monitor:
        mb_rec_before = mem_monitor.get_stats()
        perf.gpu_mem_before_recognition = mb_rec_before.used_mb if mb_rec_before else None

    rec_start = time.time()
    rec_results, rec_info = trba_predict_safe(
        recognizer, word_images, batch_sizer,
        allow_fallback_cpu=True,   # we still allow, but condition inside checks batch=1
        gpu_index=gpu_index, log_oom_cb=log_oom_cb, page_stats=page_stats,
    )
    perf.time_recognition = time.time() - rec_start
    perf.final_batch = rec_info["final_batch_size"]
    perf.batch_reductions = rec_info["batch_reductions"]
    perf.oom_occurred = rec_info["oom_occurred"]
    perf.fallback_to_cpu = not rec_info["used_cuda"]
    perf.used_cuda_final = rec_info["used_cuda"]
    perf.optimal_batch = rec_info["optimal_batch"]
    perf.predict_time_ms = rec_info["predict_time_ms"]

    if mem_monitor:
        mb_rec_after = mem_monitor.get_stats()
        perf.gpu_mem_after_recognition = mb_rec_after.used_mb if mb_rec_after else None
        perf.gpu_mem_peak = mem_monitor.peak_used_mb

    logger.info(f"[{image_name}] TRBA: {perf.time_recognition:.3f}s, final_batch={perf.final_batch}, reductions={perf.batch_reductions}, fallback={perf.fallback_to_cpu}")

    # --- Assign results and save crops ---
    for idx, (word_obj, rec, crop) in enumerate(zip(word_objects, rec_results, word_images)):
        text = rec["text"]
        conf = float(rec["confidence"])
        word_obj.text = text
        word_obj.recognition_confidence = conf
        if selftrain_dir and text:
            if conf >= conf_high:
                bucket = "high_conf"
            elif conf >= conf_mid:
                bucket = "mid_conf"
            else:
                bucket = None
            if bucket:
                save_word_for_selftrain(crop, text, conf, page_id, 0, idx, selftrain_dir, bucket)

    # --- Save final page and text ---
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

    perf.time_total = time.time() - start_total
    if perf.num_words > 0:
        perf.throughput_detection = perf.num_words / perf.time_detection if perf.time_detection > 0 else float('inf')
        perf.throughput_recognition = perf.num_words / perf.time_recognition if perf.time_recognition > 0 else float('inf')
        perf.throughput_total = perf.num_words / perf.time_total

    # Final OOM log
    if mem_monitor:
        final_mem = mem_monitor.get_stats()
        if final_mem:
            oom_writer.writerow([
                time.time(), variant, page_stats.page_id, page_stats.image_name, page_stats.image_path,
                "trba_done", 0, int(perf.fallback_to_cpu), init_batch, perf.final_batch,
                page_stats.width, page_stats.height, page_stats.num_words,
                page_stats.avg_word_area, page_stats.max_word_area, page_stats.min_word_area,
                final_mem.total_mb, final_mem.used_mb, final_mem.free_mb,
                final_mem.torch_alloc_mb, final_mem.other_mb, final_mem.peak_used_mb
            ])
    oom_csv.close()

    if record_performance:
        log_performance(perf, log_dir)

    logger.info(f"[{image_name}] Total: {perf.time_total:.3f}s, {perf.num_words} words, {perf.throughput_total:.2f} words/sec")
    return perf


def log_performance(perf: PerformanceRecord, log_dir: Path):
    perf_path = log_dir / "performance.csv"
    new_file = not perf_path.exists()
    with open(perf_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow([
                "timestamp", "variant", "page_id", "image_name", "image_path",
                "width", "height", "num_words", "avg_word_area", "max_word_area", "min_word_area",
                "time_detection", "time_cropping", "time_recognition", "time_total",
                "throughput_detection", "throughput_recognition", "throughput_total",
                "gpu_mem_before_detection", "gpu_mem_after_detection",
                "gpu_mem_before_recognition", "gpu_mem_after_recognition", "gpu_mem_peak",
                "init_batch", "final_batch", "batch_reductions", "fallback_to_cpu", "oom_occurred", "used_cuda_final",
                "optimal_batch", "convert_time_ms", "predict_time_ms"
            ])
        writer.writerow([
            perf.timestamp, perf.variant, perf.page_id, perf.image_name, perf.image_path,
            perf.width, perf.height, perf.num_words, perf.avg_word_area, perf.max_word_area, perf.min_word_area,
            f"{perf.time_detection:.3f}", f"{perf.time_cropping:.3f}", f"{perf.time_recognition:.3f}", f"{perf.time_total:.3f}",
            f"{perf.throughput_detection:.2f}", f"{perf.throughput_recognition:.2f}", f"{perf.throughput_total:.2f}",
            perf.gpu_mem_before_detection or "", perf.gpu_mem_after_detection or "",
            perf.gpu_mem_before_recognition or "", perf.gpu_mem_after_recognition or "",
            perf.gpu_mem_peak or "",
            perf.init_batch, perf.final_batch, perf.batch_reductions, int(perf.fallback_to_cpu), int(perf.oom_occurred), int(perf.used_cuda_final),
            perf.optimal_batch, f"{perf.convert_time_ms:.2f}", f"{perf.predict_time_ms:.2f}"
        ])


def iter_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".jp2"}
    return [p for p in sorted(input_path.rglob("*")) if p.is_file() and p.suffix.lower() in exts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="File or directory with images")
    parser.add_argument("--backend", type=str, default="cupy", choices=["cupy", "numpy"],
                        help="Array backend: 'cupy' (GPU) or 'numpy' (CPU)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device for EAST/TRBA models")
    parser.add_argument("--cache-dir", type=str, default="cache", help="Cache directory")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--selftrain-dir", type=str, default=None, help="Self-training crops directory")
    parser.add_argument("--init-batch", type=int, default=16, help="Initial batch size for TRBA")
    parser.add_argument("--min-batch", type=int, default=1, help="Minimum batch size (1 = never fallback to CPU)")
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU index")
    parser.add_argument("--conf-high", type=float, default=0.97, help="High confidence threshold")
    parser.add_argument("--conf-mid", type=float, default=0.90, help="Mid confidence threshold")
    parser.add_argument("--min-text-size", type=int, default=5, help="Minimum text size in pixels")
    parser.add_argument("--rotate-threshold", type=float, default=1.5, help="Rotation threshold")
    parser.add_argument("--no-perf", action="store_true", help="Disable performance logging")
    args = parser.parse_args()

    global xp, cupy_available, asnumpy
    xp, cupy_available, asnumpy = import_backend(args.backend)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    input_path = Path(args.input)
    cache_dir = Path(args.cache_dir)
    log_dir = Path(args.log_dir)
    selftrain_dir = Path(args.selftrain_dir) if args.selftrain_dir else None

    logger.info(f"Initialising EAST(device={args.device}) and TRBA(device={args.device})")
    detector = EAST(device=args.device)
    recognizer = TRBA(weights="trba_base_g1", device=args.device)

    batch_sizer = AdaptiveBatchSizer(
        init_batch=args.init_batch,
        min_batch=args.min_batch,
        increase_factor=1.2,
        decrease_factor=0.7,
        stable_rounds=2
    )

    images = iter_images(input_path)
    logger.info(f"Found {len(images)} images. Initial batch = {args.init_batch}")

    for img in images:
        try:
            process_image(
                image_path=img, detector=detector, recognizer=recognizer, batch_sizer=batch_sizer,
                cache_dir=cache_dir, log_dir=log_dir, selftrain_dir=selftrain_dir, variant=args.backend,
                init_batch=args.init_batch, min_batch=args.min_batch, gpu_index=args.gpu_index,
                conf_high=args.conf_high, conf_mid=args.conf_mid,
                min_text_size=args.min_text_size, rotate_threshold=args.rotate_threshold,
                record_performance=not args.no_perf
            )
        except Exception as e:
            logger.exception(f"Error processing {img.name}: {e}")


if __name__ == "__main__":
    main()
