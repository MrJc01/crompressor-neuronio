#!/usr/bin/env python3
"""
COLAB NOTEBOOK — Lab05 Real: Tree of Thoughts com GPT-2 Real
Compara geração autoregressiva vs ToT com LLM real.

INSTRUÇÕES PARA COLAB:
1. Abra https://colab.research.google.com
2. Runtime → Change runtime type → T4 GPU
3. Cole este script inteiro em uma célula e execute
4. Copie o JSON de resultado e cole no chat
"""

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "transformers", "accelerate", "torch"])

import torch
import numpy as np
import json
import time
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")

# ── Carregar GPT-2 ────────────────────────────────────────────
print("\n▶ Carregando GPT-2...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
model.eval()
tokenizer.pad_token = tokenizer.eos_token
print(f"  Modelo: GPT-2 ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")

# ── Funções de geração ────────────────────────────────────────

def generate_autoregressivo(prompt, max_tokens=50, temperature=0.8):
    """Geração autoregressiva padrão (greedy/sampling)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text[len(prompt):].strip()

def generate_tot(prompt, n_branches=5, max_tokens=50, temperature=1.0):
    """Tree of Thoughts: gerar N branches e selecionar a melhor."""
    candidates = []
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    for _ in range(n_branches):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        continuation = text[len(prompt):].strip()

        # Avaliar qualidade: log-probability média
        with torch.no_grad():
            eval_inputs = tokenizer(text, return_tensors="pt").to(device)
            eval_outputs = model(**eval_inputs, labels=eval_inputs["input_ids"])
            loss = eval_outputs.loss.item()  # Negative log-likelihood

        candidates.append({
            "text": continuation,
            "loss": loss,
            "score": -loss,  # Maior = melhor
        })

    # Selecionar melhor (menor loss = maior probabilidade)
    best = max(candidates, key=lambda c: c["score"])
    return best["text"], candidates

# ── Benchmark ─────────────────────────────────────────────────
print("\n▶ Rodando benchmark...")

prompts = [
    "The theory of relativity states that",
    "In machine learning, overfitting occurs when",
    "The capital of France is Paris, which is known for",
    "Quantum computing uses qubits that can be",
    "Neural networks learn by adjusting their",
    "The process of photosynthesis converts",
    "Climate change is caused primarily by",
    "The human brain contains approximately",
    "Artificial intelligence was first proposed by",
    "The speed of light in vacuum is",
]

results_auto = []
results_tot = []

for i, prompt in enumerate(prompts):
    print(f"\n  Prompt {i+1}/{len(prompts)}: {prompt[:50]}...")

    # Autoregressivo
    t0 = time.time()
    text_auto = generate_autoregressivo(prompt, max_tokens=40)
    time_auto = time.time() - t0

    # Avaliar perplexity do resultado
    with torch.no_grad():
        full_auto = prompt + " " + text_auto
        inputs_a = tokenizer(full_auto, return_tensors="pt").to(device)
        out_a = model(**inputs_a, labels=inputs_a["input_ids"])
        loss_auto = out_a.loss.item()

    results_auto.append({
        "prompt": prompt,
        "text": text_auto[:100],
        "loss": round(loss_auto, 4),
        "perplexity": round(np.exp(loss_auto), 2),
        "time_s": round(time_auto, 3),
        "tokens": len(tokenizer.encode(text_auto)),
    })

    # Tree of Thoughts (5 branches)
    t0 = time.time()
    text_tot, candidates = generate_tot(prompt, n_branches=5, max_tokens=40)
    time_tot = time.time() - t0

    with torch.no_grad():
        full_tot = prompt + " " + text_tot
        inputs_t = tokenizer(full_tot, return_tensors="pt").to(device)
        out_t = model(**inputs_t, labels=inputs_t["input_ids"])
        loss_tot = out_t.loss.item()

    results_tot.append({
        "prompt": prompt,
        "text": text_tot[:100],
        "loss": round(loss_tot, 4),
        "perplexity": round(np.exp(loss_tot), 2),
        "time_s": round(time_tot, 3),
        "n_branches": 5,
        "best_of": len(candidates),
    })

    print(f"    Auto: ppl={np.exp(loss_auto):.1f} ({time_auto:.2f}s)")
    print(f"    ToT:  ppl={np.exp(loss_tot):.1f} ({time_tot:.2f}s) [melhor de {len(candidates)}]")

# ── Análise ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Lab05 Real — ToT vs Autoregressivo (GPT-2)")
print(f"{'='*60}")

avg_ppl_auto = np.mean([r["perplexity"] for r in results_auto])
avg_ppl_tot = np.mean([r["perplexity"] for r in results_tot])
avg_time_auto = np.mean([r["time_s"] for r in results_auto])
avg_time_tot = np.mean([r["time_s"] for r in results_tot])

tot_wins = sum(1 for a, t in zip(results_auto, results_tot) if t["perplexity"] < a["perplexity"])
melhoria_ppl = (1 - avg_ppl_tot / avg_ppl_auto) * 100

print(f"  Autoregressivo: ppl={avg_ppl_auto:.1f}, tempo={avg_time_auto:.2f}s")
print(f"  ToT (5 branches): ppl={avg_ppl_tot:.1f}, tempo={avg_time_tot:.2f}s")
print(f"  ToT venceu: {tot_wins}/{len(prompts)} prompts")
print(f"  Melhoria perplexity: {melhoria_ppl:.1f}%")
print(f"  Overhead tempo: {avg_time_tot/avg_time_auto:.1f}x")

# ── JSON ──────────────────────────────────────────────────────
final_result = {
    "lab": "lab05_real",
    "modelo": "gpt2",
    "device": device,
    "n_prompts": len(prompts),
    "resumo": {
        "avg_ppl_auto": round(avg_ppl_auto, 2),
        "avg_ppl_tot": round(avg_ppl_tot, 2),
        "melhoria_ppl_pct": round(melhoria_ppl, 1),
        "tot_wins": tot_wins,
        "overhead_tempo": round(avg_time_tot / avg_time_auto, 1),
    },
    "autoregressivo": results_auto,
    "tot": results_tot,
}

print(f"\n{'='*60}")
print("  COPIE O JSON ABAIXO E COLE NO CHAT:")
print(f"{'='*60}")
print(json.dumps(final_result, indent=2, ensure_ascii=False))
