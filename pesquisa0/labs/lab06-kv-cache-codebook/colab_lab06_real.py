# Lab06 Real — KV Cache Codebook com GPT-2 (Google Colab)
#
# Instruções:
# 1. Abra https://colab.research.google.com
# 2. Crie um novo notebook
# 3. Copie e cole cada célula abaixo
# 4. Runtime → Change runtime type → T4 GPU
# 5. Execute todas as células em ordem

# ─── Célula 1: Instalação ─────────────────────────────────────
# !pip install -q transformers accelerate torch numpy

# ─── Célula 2: Importações ────────────────────────────────────
"""
import torch
import numpy as np
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
"""

# ─── Célula 3: Carregar GPT-2 ─────────────────────────────────
"""
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

print(f"Modelo: {model_name}")
print(f"Parâmetros: {sum(p.numel() for p in model.parameters()):,}")
print(f"Config: {model.config.n_head} heads, {model.config.n_embd} dim")
"""

# ─── Célula 4: Extrair KV Cache Real ──────────────────────────
"""
# Texto de teste (512 tokens)
test_text = "The history of artificial intelligence began in antiquity, " * 50
inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512).to(device)

with torch.no_grad():
    outputs = model(**inputs, use_cache=True)
    kv_cache = outputs.past_key_values

# kv_cache é uma tupla de (key, value) para cada camada
# Cada tensor: (batch=1, n_heads=12, seq_len, head_dim=64)
n_layers = len(kv_cache)
seq_len = kv_cache[0][0].shape[2]
n_heads = kv_cache[0][0].shape[1]
head_dim = kv_cache[0][0].shape[3]

print(f"Camadas: {n_layers}")
print(f"Seq len: {seq_len}")
print(f"Heads: {n_heads}, Head dim: {head_dim}")

# Calcular memória total do KV Cache
mem_total = sum(k.nbytes + v.nbytes for k, v in kv_cache)
print(f"Memória KV Cache: {mem_total / 1024:.1f} KB")
"""

# ─── Célula 5: Vector Quantization do KV Cache ────────────────
"""
def quantize_kv_layer(tensor, K=256):
    '''Aplica VQ a um tensor de KV Cache (1, heads, seq, dim).'''
    batch, heads, seq, dim = tensor.shape
    flat = tensor.reshape(-1, dim).cpu().numpy()  # (heads*seq, dim)
    N = flat.shape[0]

    # K-Means simplificado (3 iterações)
    idx_init = np.random.choice(N, K, replace=False)
    codebook = flat[idx_init].copy()

    for iteration in range(3):
        # Atribuir
        A2 = np.sum(flat**2, axis=1, keepdims=True)
        B2 = np.sum(codebook**2, axis=1)
        dists = A2 + B2 - 2 * np.dot(flat, codebook.T)
        indices = np.argmin(dists, axis=1).astype(np.int16)

        # Atualizar centroids
        for k in range(K):
            mask = indices == k
            if mask.sum() > 0:
                codebook[k] = flat[mask].mean(axis=0)

    # Reconstruir
    reconstructed = codebook[indices].reshape(batch, heads, seq, dim)
    mse = float(np.mean((flat - codebook[indices])**2))

    # Tamanhos
    mem_original = tensor.nbytes
    mem_indices = indices.nbytes
    mem_codebook = codebook.nbytes
    mem_compressed = mem_indices + mem_codebook

    return {
        "mse": mse,
        "mem_original": mem_original,
        "mem_compressed": mem_compressed,
        "reduction_pct": round((1 - mem_compressed/mem_original) * 100, 2),
        "ratio": round(mem_original / mem_compressed, 2),
        "reconstructed": torch.tensor(reconstructed).to(tensor.device)
    }

# Quantizar todas as camadas
results_by_k = {}
for K in [64, 128, 256, 512]:
    total_mse = 0
    total_orig = 0
    total_comp = 0

    t0 = time.time()
    for layer_idx, (k_cache, v_cache) in enumerate(kv_cache):
        res_k = quantize_kv_layer(k_cache, K)
        res_v = quantize_kv_layer(v_cache, K)
        total_mse += res_k["mse"] + res_v["mse"]
        total_orig += res_k["mem_original"] + res_v["mem_original"]
        total_comp += res_k["mem_compressed"] + res_v["mem_compressed"]
    t_total = time.time() - t0

    avg_mse = total_mse / (n_layers * 2)
    reduction = (1 - total_comp/total_orig) * 100
    ratio = total_orig / total_comp

    results_by_k[K] = {
        "K": K, "avg_mse": round(avg_mse, 6),
        "total_orig_KB": round(total_orig/1024, 1),
        "total_comp_KB": round(total_comp/1024, 1),
        "reduction_pct": round(reduction, 1),
        "ratio": round(ratio, 1),
        "time_s": round(t_total, 2)
    }
    print(f"K={K:>4}: {reduction:.1f}% redução ({ratio:.1f}x) | MSE={avg_mse:.6f} | {t_total:.1f}s")

print("\\nResultados prontos! Copie o JSON abaixo:")
print(json.dumps(results_by_k, indent=2))
"""

# ─── Célula 6: Medir Perplexity ───────────────────────────────
"""
def compute_perplexity(model, tokenizer, text, max_length=512):
    '''Calcula perplexity de um texto.'''
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return float(torch.exp(outputs.loss))

# Textos de teste
test_texts = [
    "The quick brown fox jumps over the lazy dog. " * 20,
    "In the beginning was the Word, and the Word was with God. " * 20,
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2) " * 20,
]

print("Perplexity com KV Cache ORIGINAL:")
for i, text in enumerate(test_texts):
    ppl = compute_perplexity(model, tokenizer, text)
    print(f"  Texto {i+1}: {ppl:.2f}")

print("\\n(Para comparar com KV Cache quantizado,")
print("seria necessário modificar o attention do modelo.")
print("A medição de MSE já dá uma proxy da qualidade.)")
"""

# ─── Célula 7: Salvar Resultados ──────────────────────────────
"""
# Copie o output do JSON da Célula 5 e salve como:
# pesquisa0/resultados/lab06_colab_results.json
print("Copie os resultados e cole no arquivo local!")
"""
