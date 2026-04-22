# Lab06 Real — KV Cache Codebook com GPT-2 (Google Colab)
#
# INSTRUÇÕES:
# 1. Abra https://colab.research.google.com
# 2. Crie um novo notebook
# 3. Runtime → Change runtime type → T4 GPU  ← IMPORTANTE!
# 4. Copie CADA CÉLULA abaixo (entre os marcadores ═══)
#    para células separadas no Colab
# 5. Execute em ordem

# ═══════════════════════════════════════════════════════════════
# CÉLULA 1: Instalação
# ═══════════════════════════════════════════════════════════════
"""
!pip install -q transformers accelerate torch numpy
"""

# ═══════════════════════════════════════════════════════════════
# CÉLULA 2: Importações e Setup
# ═══════════════════════════════════════════════════════════════
"""
import torch
import numpy as np
import json
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  SEM GPU! Vá em Runtime → Change runtime type → T4 GPU")
    print("   O script roda em CPU mas será mais lento.")
"""

# ═══════════════════════════════════════════════════════════════
# CÉLULA 3: Carregar GPT-2
# ═══════════════════════════════════════════════════════════════
"""
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

print(f"Modelo: {model_name}")
print(f"Parâmetros: {sum(p.numel() for p in model.parameters()):,}")
print(f"Config: {model.config.n_head} heads, {model.config.n_embd} dim, head_dim={model.config.n_embd // model.config.n_head}")
"""

# ═══════════════════════════════════════════════════════════════
# CÉLULA 4: Extrair KV Cache Real
# ═══════════════════════════════════════════════════════════════
"""
# Texto de teste (~512 tokens)
test_text = "The history of artificial intelligence began in antiquity, with myths and stories of artificial beings. The seeds of modern AI were planted by philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s. " * 5
inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512).to(device)

with torch.no_grad():
    outputs = model(**inputs, use_cache=True)
    past = outputs.past_key_values

# API nova do transformers: DynamicCache
# Extrair tensores de cada camada
kv_tensors = []
if hasattr(past, 'key_cache'):
    # DynamicCache (transformers >= 4.36)
    n_layers = len(past.key_cache)
    for i in range(n_layers):
        k = past.key_cache[i]   # (batch, heads, seq, dim)
        v = past.value_cache[i] # (batch, heads, seq, dim)
        kv_tensors.append((k, v))
    print(f"API: DynamicCache (novo)")
else:
    # Tupla clássica (transformers < 4.36)
    n_layers = len(past)
    for k, v in past:
        kv_tensors.append((k, v))
    print(f"API: Tuple (clássico)")

k0, v0 = kv_tensors[0]
seq_len = k0.shape[2]
n_heads = k0.shape[1]
head_dim = k0.shape[3]

print(f"Camadas: {n_layers}")
print(f"Seq len: {seq_len}")
print(f"Heads: {n_heads}, Head dim: {head_dim}")

# Calcular memória total do KV Cache
mem_total = sum(k.element_size() * k.nelement() + v.element_size() * v.nelement() for k, v in kv_tensors)
print(f"Memória KV Cache: {mem_total / 1024:.1f} KB ({mem_total / 1e6:.2f} MB)")
"""

# ═══════════════════════════════════════════════════════════════
# CÉLULA 5: Vector Quantization do KV Cache
# ═══════════════════════════════════════════════════════════════
"""
def quantize_tensor(tensor, K=256, n_iter=3):
    '''Aplica Vector Quantization a um tensor (batch, heads, seq, dim).'''
    b, h, s, d = tensor.shape
    flat = tensor.reshape(-1, d).cpu().float().numpy()
    N = flat.shape[0]

    # K-Means (n_iter iterações)
    np.random.seed(42)
    idx_init = np.random.choice(N, min(K, N), replace=False)
    codebook = flat[idx_init].copy()

    for it in range(n_iter):
        # Atribuir cada vetor ao centroid mais próximo
        # (A-B)^2 = A^2 + B^2 - 2*A*B
        A2 = np.sum(flat**2, axis=1, keepdims=True)
        B2 = np.sum(codebook**2, axis=1)
        dists = A2 + B2 - 2 * flat @ codebook.T
        indices = np.argmin(dists, axis=1).astype(np.int16)

        # Atualizar centroids
        for k in range(codebook.shape[0]):
            mask = (indices == k)
            if mask.sum() > 0:
                codebook[k] = flat[mask].mean(axis=0)

    # Métricas
    reconstructed = codebook[indices]
    mse = float(np.mean((flat - reconstructed)**2))
    cosine_sims = []
    for i in range(min(1000, N)):
        a, b_vec = flat[i], reconstructed[i]
        cos = np.dot(a, b_vec) / (np.linalg.norm(a) * np.linalg.norm(b_vec) + 1e-10)
        cosine_sims.append(cos)
    avg_cosine = float(np.mean(cosine_sims))

    mem_orig = tensor.element_size() * tensor.nelement()
    mem_idx = indices.nbytes
    mem_cb = codebook.nbytes
    mem_comp = mem_idx + mem_cb

    return {
        "mse": round(mse, 8),
        "cosine_sim": round(avg_cosine, 6),
        "mem_original": mem_orig,
        "mem_compressed": mem_comp,
        "reduction_pct": round((1 - mem_comp/mem_orig) * 100, 2),
        "ratio": round(mem_orig / mem_comp, 2)
    }


# Rodar quantização para vários K
print("="*60)
print("  RESULTADOS: Vector Quantization do KV Cache GPT-2")
print("="*60)

all_results = {}
for K in [64, 128, 256, 512]:
    total_mse = 0
    total_cosine = 0
    total_orig = 0
    total_comp = 0
    count = 0

    t0 = time.time()
    for layer_idx, (k_cache, v_cache) in enumerate(kv_tensors):
        res_k = quantize_tensor(k_cache, K)
        res_v = quantize_tensor(v_cache, K)
        total_mse += res_k["mse"] + res_v["mse"]
        total_cosine += res_k["cosine_sim"] + res_v["cosine_sim"]
        total_orig += res_k["mem_original"] + res_v["mem_original"]
        total_comp += res_k["mem_compressed"] + res_v["mem_compressed"]
        count += 2
    t_total = time.time() - t0

    avg_mse = total_mse / count
    avg_cosine = total_cosine / count
    reduction = (1 - total_comp/total_orig) * 100
    ratio = total_orig / total_comp

    all_results[str(K)] = {
        "K": K,
        "avg_mse": round(avg_mse, 8),
        "avg_cosine_similarity": round(avg_cosine, 6),
        "total_original_KB": round(total_orig/1024, 1),
        "total_compressed_KB": round(total_comp/1024, 1),
        "reduction_pct": round(reduction, 1),
        "ratio": round(ratio, 1),
        "time_seconds": round(t_total, 2)
    }
    print(f"  K={K:>4}: {reduction:.1f}% redução ({ratio:.1f}x) | MSE={avg_mse:.8f} | Cosine={avg_cosine:.4f} | {t_total:.1f}s")

print()
print("="*60)
print("  JSON PARA COPIAR:")
print("="*60)
final_json = {
    "meta": {
        "model": model_name,
        "params": sum(p.numel() for p in model.parameters()),
        "seq_len": seq_len,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "device": device,
    },
    "results": all_results
}
print(json.dumps(final_json, indent=2))
"""

# ═══════════════════════════════════════════════════════════════
# CÉLULA 6: Perplexity (opcional)
# ═══════════════════════════════════════════════════════════════
"""
def compute_perplexity(text, max_length=256):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return float(torch.exp(outputs.loss))

test_texts = {
    "prosa": "The quick brown fox jumps over the lazy dog. This is a simple test of language model perplexity. " * 10,
    "codigo": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2). def factorial(n): return 1 if n == 0 else n * factorial(n-1). " * 10,
    "tecnico": "The transformer architecture uses multi-head self-attention mechanisms to process sequential data in parallel. Each attention head learns different aspects of the input relationships. " * 10,
}

print("Perplexity do modelo original (referência):")
for nome, text in test_texts.items():
    ppl = compute_perplexity(text)
    print(f"  {nome:>10}: {ppl:.2f}")
"""
