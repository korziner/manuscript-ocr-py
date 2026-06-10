#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import re
import time
import torch
import gc
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback
)
from trl import SFTTrainer, SFTConfig

# ========== Muon optimiser (fixed) ==========
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
                p.data.add_(buf, alpha=-lr)   # fixed in-place

# ========== QAT (torchao) ==========
try:
    from torchao.quantization import quantize_
    from torchao.quantization.qat import QATConfig
    from torchao.quantization import Int8DynamicActivationInt4WeightConfig
    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False

# ========== CORRECT BENCHMARK (with synchronisation) ==========
def benchmark_matmul_correct(device='cuda', size=4096, duration=1.0, dtype=torch.float16):
    """Точный замер matmul с синхронизацией через torch.cuda.Event."""
    a = torch.randn(size, size, dtype=dtype, device=device)
    b = torch.randn(size, size, dtype=dtype, device=device)
    # Прогрев (синхронизированный)
    for _ in range(10):
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
    # Измерение
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    iters = 0
    while True:
        c = torch.matmul(a, b)
        iters += 1
        if iters % 10 == 0:
            torch.cuda.synchronize()
        end_event.record()
        torch.cuda.synchronize()
        if start_event.elapsed_time(end_event) / 1000.0 >= duration:
            break
    elapsed = start_event.elapsed_time(end_event) / 1000.0
    flops = 2 * size * size * size * iters
    tflops = flops / elapsed / 1e12
    return tflops

def extended_benchmark(device='cuda', duration=0.8):
    """Размеры: важные для tiling, L2 кэша (5MB) и typical hidden sizes."""
    # Интересующие размеры: 384, 512, 640, 768, 896, 1024, 1152, 1280, 1536, 2048, 2560, 3072, 4096
    sizes = [384, 512, 640, 768, 896, 1024, 1152, 1280, 1536, 2048, 2560, 3072, 4096]
    dtypes = [torch.float16, torch.bfloat16, torch.float32]
    print("\n🧪 Корректный бенчмарк (синхронизированный, с разными размерами):")
    for dtype in dtypes:
        print(f"\n--- {dtype} ---")
        for size in sizes:
            if dtype == torch.float32 and size > 4096:
                continue
            tflops = benchmark_matmul_correct(device=device, size=size, duration=duration, dtype=dtype)
            print(f"size={size:4d} -> {tflops:5.2f} TFLOP/s")
    print()



# ========== Data loading ==========
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

# ========== Callbacks ==========
class WeightsOnlyCheckpointCallback(TrainerCallback):
    def __init__(self, output_dir, save_steps):
        self.output_dir = output_dir
        self.save_steps = save_steps
        os.makedirs(output_dir, exist_ok=True)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step % self.save_steps == 0 and step > 0:
            ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            if state.log_history:
                with open(os.path.join(ckpt_dir, "trainer_state.json"), "w") as f:
                    json.dump({"global_step": step, "log_history": state.log_history[-100:]}, f)
            print(f"\n💾 Checkpoint saved (weights only): {ckpt_dir}")
            torch.cuda.empty_cache()
            gc.collect()

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

# ========== Argument parsing ==========
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Gemma 3 270M with Muon + QAT + weights‑only checkpoints",
        epilog="""
Examples:
  # Basic training with all benchmarks
  python train.py --data_path ./data.json

  # Skip benchmarks (faster startup)
  python train.py --skip_benchmarks

  # Resume from a checkpoint
  python train.py --resume_from ./checkpoint-500

  # Inference only
  python train.py --inference_only --resume_from ./checkpoint-500 --prompt "Hello"

  # Disable Muon and QAT (use AdamW, no quantization)
  python train.py --disable_muon --disable_qat

Benchmark results (TFLOPS) are used to:
  - Estimate training speed (tokens/sec)
  - Choose optimal batch size / sequence length
  - Detect thermal throttling or low-power mode
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_path", type=str, default=r"/content/training_data_19century.json",
                        help="Path to JSONL training data")
    parser.add_argument("--model_name", type=str, default="oopere/gemma-3-270m-14L-distilled",
                        help="HuggingFace model name")
    parser.add_argument("--output_dir", type=str, default="model_muon_qat_checkpoints",
                        help="Directory for checkpoints and final model")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume training from a weights‑only checkpoint (e.g. ./checkpoint-500)")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save a checkpoint every N steps")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per‑device batch size")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max sequence length (tokens)")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 mixed precision (default: enabled)")
    parser.add_argument("--no_fp16", dest="fp16", action="store_false",
                        help="Disable FP16")
    parser.add_argument("--skip_benchmarks", action="store_true",
                        help="Skip GPU benchmarks (faster startup)")
    parser.add_argument("--inference_only", action="store_true",
                        help="Run inference only, no training")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt for inference mode")
    parser.add_argument("--disable_qat", action="store_true",
                        help="Disable QAT (no quantization during training)")
    parser.add_argument("--disable_muon", action="store_true",
                        help="Disable Muon, use AdamW instead")
    return parser.parse_args()

# ========== Main ==========
def main():
    args = parse_args()

    # Load data
    print(f"📥 Loading data from {args.data_path}...")
    dataset = load_data(args.data_path)
    print(f"✅ Loaded {len(dataset)} examples")

    # Load tokenizer and model
    print(f"📦 Loading model {args.model_name}...")
    if args.resume_from and not args.inference_only:
        print(f"🔄 Resuming from checkpoint: {args.resume_from}")
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

    # Benchmarks (optional)
    if not args.skip_benchmarks and not args.inference_only:
        print("\n🔬 Single matmul (4096x4096 FP16):")
        tflops = benchmark_matmul_correct(size=4096, duration=1.0)
        print(f"✅ Performance: {tflops:.2f} TFLOP/s")
        if tflops >= 40:
            print("🎉 Tensor Cores running at full speed")
        else:
            print(f"⚠️ Low performance: {tflops:.2f} TFLOP/s (check power/thermal)")

        extended_benchmark(duration=0.8)   # detailed sizes
    else:
        print("\n⏩ Skipping benchmarks (--skip_benchmarks)")

    # QAT prepare
    if TORCHAO_AVAILABLE and not args.disable_qat and not args.inference_only:
        qat_config = Int8DynamicActivationInt4WeightConfig(group_size=32)
        quantize_(model, QATConfig(qat_config, step="prepare"))
        print("✅ QAT prepare applied")

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

    # Training config (disable internal saving)
    training_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        packing=False,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        fp16=args.fp16,
        bf16=False,
        logging_steps=50,
        save_steps=1_000_000,       # disable built-in saving
        save_total_limit=1,
        dataset_text_field="text",
        report_to="none",
        remove_unused_columns=False,
    )

    # Optimizer
    if not args.disable_muon:
        optimizer = Muon(model.parameters(), lr=5e-7, momentum=0.95, weight_decay=0.01)   

       #optimizer = Muon(model.parameters(), lr=args.learning_rate, momentum=0.95, weight_decay=0.01)
        print("🔧 Using Muon optimizer")
    else:
        optimizer = None
        print("🔧 Using AdamW (default)")

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        optimizers=(optimizer, None) if optimizer else (None, None),
        callbacks=[
            WeightsOnlyCheckpointCallback(args.output_dir, args.save_steps),
            InferenceCallback(tokenizer, ["Какие три основных цвѣта?", "Кто ты?"], every=500)
        ]
    )

    print("\n🚀 Starting training...")
    trainer.train()

    # Final save + QAT convert
    print("\n💾 Saving final model...")
    if TORCHAO_AVAILABLE and not args.disable_qat:
        quantize_(model, QATConfig(qat_config, step="convert"))
        print("✅ QAT convert applied")
    trainer.save_model(f"{args.output_dir}_final")
    tokenizer.save_pretrained(f"{args.output_dir}_final")
    print(f"✅ Training finished. Model saved to {args.output_dir}_final")

if __name__ == "__main__":
    main()
