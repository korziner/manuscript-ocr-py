#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
if "--help" in sys.argv or "-h" in sys.argv:
    print("Использование: python NorMuon_hybrid.py [опции]")
    print("  --use_normuon         Использовать гибридный оптимизатор (NorMuon для матриц, AdamW для векторов)")
    print("  --disable_muon        Использовать AdamW для всех параметров")
    print("  --learning_rate LR    Скорость обучения (для NorMuon и AdamW одинаковая, можно задать отдельно через --lr_matrix и --lr_vector)")
    print("  --lr_matrix LR        Скорость для NorMuon (по умолч. 5e-7)")
    print("  --lr_vector LR        Скорость для AdamW (по умолч. 2e-7)")
    print("  --output_dir DIR      ... и т.д. Полный список опций см. в аргументах.")
    sys.exit(0)

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import re
import time
import torch
import gc
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import SFTTrainer, SFTConfig

# ---------- 1. Muon (оригинальный) ----------
def zeropower_via_newtonschulz5(G, steps=5):
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.01):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            closure()
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    g = g.add(p, alpha=wd)
                if p.ndim >= 2:
                    g = zeropower_via_newtonschulz5(g)
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                p.data.add_(buf, alpha=-lr)

# ---------- 2. Гибридный оптимизатор (NorMuon + AdamW) ----------
try:
    from dion import NorMuon
    DION_AVAILABLE = True
except ImportError:
    DION_AVAILABLE = False
    print("⚠️ dion не установлен. Установите: pip install git+https://github.com/microsoft/dion.git")


class HybridOptimizer(torch.optim.Optimizer):
    def __init__(self, matrix_params, vector_params, lr_matrix=5e-7, lr_vector=2e-7, betas=(0.9, 0.95), weight_decay=0.01):
        # Создаём внутренние оптимизаторы
        self.matrix_optim = NorMuon(matrix_params, lr=lr_matrix, betas=betas, weight_decay=weight_decay)
        self.vector_optim = torch.optim.AdamW(vector_params, lr=lr_vector, weight_decay=weight_decay)
        # Объединяем param_groups для совместимости
        self.param_groups = self.matrix_optim.param_groups + self.vector_optim.param_groups
        self.defaults = {}

    def step(self, closure=None):
        self.matrix_optim.step(closure)
        self.vector_optim.step(closure)

    def state_dict(self):
        return {'matrix': self.matrix_optim.state_dict(), 'vector': self.vector_optim.state_dict()}

    def load_state_dict(self, state_dict):
        self.matrix_optim.load_state_dict(state_dict['matrix'])
        self.vector_optim.load_state_dict(state_dict['vector'])

    def zero_grad(self, set_to_none=True):
        self.matrix_optim.zero_grad(set_to_none)
        self.vector_optim.zero_grad(set_to_none)

    # Дополнительные методы, которые могут понадобиться:
    def add_param_group(self, group):
        raise NotImplementedError("HybridOptimizer does not support adding param groups after init")


# ---------- 3. Данные ----------
def load_data(data_path):
    def remove_system_prompt(text):
        return re.sub(r'<\|system\|>.*?<\|user\|>', '<|user|>', text, flags=re.DOTALL)
    data_list = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                if 'text' in obj and obj['text'].strip():
                    cleaned = remove_system_prompt(obj['text']).strip()
                    if cleaned:
                        data_list.append({"text": cleaned})
            except:
                continue
    return Dataset.from_dict({"text": [item['text'] for item in data_list]})

# ---------- 4. Callbacks ----------
class WeightsOnlyCheckpointCallback(TrainerCallback):
    def __init__(self, output_dir, save_steps, tokenizer, min_free_gb=2):
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.tokenizer = tokenizer
        self.min_free_gb = min_free_gb
        os.makedirs(output_dir, exist_ok=True)

    def _free_disk_space(self, required_gb):
        import shutil
        free = shutil.disk_usage(self.output_dir).free // (2**30)
        if free >= required_gb:
            return True
        ckpts = [d for d in os.listdir(self.output_dir) if d.startswith("checkpoint-")]
        ckpts.sort(key=lambda x: os.path.getmtime(os.path.join(self.output_dir, x)))
        for ckpt in ckpts:
            if free >= required_gb:
                break
            path = os.path.join(self.output_dir, ckpt)
            shutil.rmtree(path, ignore_errors=True)
            free = shutil.disk_usage(self.output_dir).free // (2**30)
        return free >= required_gb

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step % self.save_steps == 0 and step > 0:
            torch.cuda.empty_cache()
            gc.collect()
            if not self._free_disk_space(self.min_free_gb):
                print(f"❌ Disk full, skipping checkpoint {step}")
                return control
            ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            try:
                model.save_pretrained(ckpt_dir)
                self.tokenizer.save_pretrained(ckpt_dir)
                if state.log_history:
                    with open(os.path.join(ckpt_dir, "trainer_state.json"), "w") as f:
                        json.dump({"global_step": step, "log_history": state.log_history[-100:]}, f)
                print(f"\n💾 Checkpoint saved: {ckpt_dir}")
            except Exception as e:
                print(f"⚠️ Failed to save checkpoint: {e}")
            torch.cuda.empty_cache()
            gc.collect()
        return control

class InferenceCallback(TrainerCallback):
    def __init__(self, tokenizer, prompts, every=500):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.every = every

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step % self.every == 0 and step > 0:
            print(f"\n🧪 [Step {step}] Test inference:")
            model.eval()
            for prompt in self.prompts:
                full_prompt = f"<|user|>\n{prompt}\n<|assistant|>"
                inputs = self.tokenizer(full_prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.7)
                response = self.tokenizer.decode(out[0], skip_special_tokens=True)
                if full_prompt in response:
                    response = response.split(full_prompt)[-1].strip()
                print(f"Q: {prompt}\nA: {response}\n")
            model.train()
            torch.cuda.empty_cache()

# ---------- 5. Парсинг аргументов ----------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=r"/content/training_data_19century.json")
    parser.add_argument("--model_name", type=str, default="oopere/gemma-3-270m-14L-distilled")
    parser.add_argument("--output_dir", type=str, default="./model_checkpoints")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-7)
    parser.add_argument("--lr_matrix", type=float, default=5e-7, help="LR for NorMuon (if --use_normuon)")
    parser.add_argument("--lr_vector", type=float, default=2e-7, help="LR for AdamW on vectors (if --use_normuon)")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no_fp16", dest="fp16", action="store_false")
    parser.add_argument("--skip_benchmarks", action="store_true")
    parser.add_argument("--inference_only", action="store_true")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--disable_qat", action="store_true")
    parser.add_argument("--disable_muon", action="store_true")
    parser.add_argument("--use_normuon", action="store_true", help="Use HybridOptimizer (NorMuon matrices + AdamW vectors)")
    return parser.parse_args()

# ---------- 6. Main ----------
def main():
    args = parse_args()

    # Data
    print(f"📥 Loading data from {args.data_path}...")
    dataset = load_data(args.data_path)
    print(f"✅ Loaded {len(dataset)} examples")

    # Model & tokenizer
    print(f"📦 Loading model {args.model_name}...")
    if args.resume_from and not args.inference_only:
        tokenizer = AutoTokenizer.from_pretrained(args.resume_from, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.resume_from, trust_remote_code=True).to("cuda")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True).to("cuda")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print(f"✅ Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Benchmarks (skip)
    if not args.skip_benchmarks:
        print("⏩ Benchmarks disabled (--skip_benchmarks)")
    else:
        print("\n⏩ Skipping benchmarks")

    # QAT (optional)
    if not args.disable_qat and not args.inference_only:
        try:
            from torchao.quantization import quantize_
            from torchao.quantization.qat import QATConfig
            from torchao.quantization import Int8DynamicActivationInt4WeightConfig
            qat_config = Int8DynamicActivationInt4WeightConfig(group_size=32)
            quantize_(model, QATConfig(qat_config, step="prepare"))
            print("✅ QAT prepare applied")
        except ImportError:
            print("⚠️ torchao not installed, skipping QAT")

    # Inference only
    if args.inference_only:
        if not args.resume_from:
            print("Error: --inference_only requires --resume_from")
            return
        prompt = args.prompt or "Какие три основных цвѣта?"
        print(f"\n🔮 Inference from {args.resume_from}")
        inputs = tokenizer(f"<|user|>\n{prompt}\n<|assistant|>", return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7)
        print(tokenizer.decode(out[0], skip_special_tokens=True))
        return

    # ---------- Optimizer selection ----------
    if args.use_normuon and DION_AVAILABLE:
        matrix_params = [p for p in model.parameters() if p.ndim >= 2]
        vector_params = [p for p in model.parameters() if p.ndim < 2]
        print(f"📊 Matrices: {len(matrix_params)} parameters, Vectors: {len(vector_params)} parameters")
        optimizer = HybridOptimizer(
            matrix_params, vector_params,
            lr_matrix=args.lr_matrix,
            lr_vector=args.lr_vector,
            betas=(0.9, 0.95),
            weight_decay=0.01
        )
        print(f"🔧 Using HybridOptimizer (NorMuon lr={args.lr_matrix}, AdamW lr={args.lr_vector})")
    elif not args.disable_muon:
        optimizer = Muon(model.parameters(), lr=args.learning_rate, momentum=0.95, weight_decay=0.01)
        print(f"🔧 Using Muon optimizer (lr={args.learning_rate})")
    else:
        optimizer = None
        print(f"🔧 Using AdamW (lr={args.learning_rate})")

    # SFTConfig
    training_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        packing=False,
        learning_rate=args.learning_rate,  # будет использован только если optimizer=None
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=1.0,
        fp16=args.fp16,
        bf16=False,
        logging_steps=50,
        save_steps=1_000_000,      # отключаем встроенное сохранение
        save_total_limit=1,
        dataset_text_field="text",
        report_to="none",
        remove_unused_columns=False,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        optimizers=(optimizer, None) if optimizer else (None, None),
        callbacks=[
            WeightsOnlyCheckpointCallback(args.output_dir, args.save_steps, tokenizer),
            InferenceCallback(tokenizer, ["Какие три основных цвѣта?", "Кто ты?"], every=500)
        ]
    )

    print("\n🚀 Starting training...")
    trainer.train()

    # Final save
    print("\n💾 Saving final model...")
    if not args.disable_qat:
        try:
            from torchao.quantization import quantize_
            from torchao.quantization.qat import QATConfig
            quantize_(model, QATConfig(qat_config, step="convert"))
            print("✅ QAT convert applied")
        except:
            pass
    trainer.save_model(f"{args.output_dir}_final")
    tokenizer.save_pretrained(f"{args.output_dir}_final")
    print(f"✅ Training finished. Model saved to {args.output_dir}_final")

if __name__ == "__main__":
    main()
