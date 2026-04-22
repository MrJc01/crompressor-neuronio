#!/usr/bin/env python3
"""
LAB06 — Codebook Learning no KV Cache (v2 — leve)
===================================================
Vector Quantization aplicada ao KV Cache simulado.
Escala reduzida para rodar em CPU sem travar.

Saída: JSON em pesquisa0/resultados/lab06_results.json
"""

import json
import os
import random
import time
from datetime import datetime

SEED = 42
random.seed(SEED)
RESULTADOS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'resultados')

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def quantizar_cache(cache, K):
    """Vector Quantization via K-Means simplificado (1 iteração)."""
    seq_len, n_heads, head_dim = cache.shape
    flat = cache.reshape(-1, head_dim)
    N = flat.shape[0]

    # Inicializar codebook com amostras aleatórias
    idx_init = np.random.choice(N, K, replace=False)
    codebook = flat[idx_init].copy()

    t0 = time.perf_counter()

    # Atribuir cada vetor ao centroid mais próximo (batch)
    indices = np.zeros(N, dtype=np.int16)
    batch_size = 4096
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        batch = flat[i:end]
        # (A-B)^2 = A^2 + B^2 - 2AB
        A2 = np.sum(batch ** 2, axis=1, keepdims=True)
        B2 = np.sum(codebook ** 2, axis=1)
        dists = A2 + B2 - 2 * np.dot(batch, codebook.T)
        indices[i:end] = np.argmin(dists, axis=1)

    # Reconstruir
    reconstructed = codebook[indices].reshape(cache.shape)
    t_quant = time.perf_counter() - t0

    mse = float(np.mean((cache - reconstructed) ** 2))
    return indices.reshape(seq_len, n_heads), codebook, mse, t_quant


def main():
    print("=" * 60)
    print("  LAB06 — CODEBOOK NO KV CACHE (v2)")
    print(f"  Numpy: {HAS_NUMPY}")
    print("=" * 60)

    if not HAS_NUMPY:
        print("  ERRO: Requer numpy.")
        return

    res = {"meta": {"timestamp": datetime.now().isoformat(), "seed": SEED},
           "experimentos": {}}

    # Escala viável para CPU: 1024 tokens, 12 heads, 64 dim
    seq_len, n_heads, head_dim = 1024, 12, 64
    np.random.seed(SEED)

    # Simular KV Cache com estrutura (não puro ruído)
    base_pattern = np.random.randn(1, n_heads, head_dim).astype(np.float32)
    k_cache = (np.random.randn(seq_len, n_heads, head_dim) * 0.3 + base_pattern).astype(np.float32)
    v_cache = (np.random.randn(seq_len, n_heads, head_dim) * 0.3 + base_pattern * 0.5).astype(np.float32)

    mem_original = k_cache.nbytes + v_cache.nbytes

    print(f"\n  Config: seq={seq_len}, heads={n_heads}, dim={head_dim}")
    print(f"  Memória original: {mem_original / 1024:.1f} KB")

    resultados_k = {}
    for K in [64, 128, 256, 512]:
        print(f"\n  ▶ K={K}...", end="", flush=True)

        idx_k, cb_k, mse_k, t_k = quantizar_cache(k_cache, K)
        idx_v, cb_v, mse_v, t_v = quantizar_cache(v_cache, K)

        mem_idx = idx_k.nbytes + idx_v.nbytes
        mem_cb = cb_k.nbytes + cb_v.nbytes
        mem_comp = mem_idx + mem_cb
        reducao = (1 - mem_comp / mem_original) * 100

        resultados_k[str(K)] = {
            "K": K,
            "mse_k": round(mse_k, 6),
            "mse_v": round(mse_v, 6),
            "mse_medio": round((mse_k + mse_v) / 2, 6),
            "mem_indices_bytes": mem_idx,
            "mem_codebook_bytes": mem_cb,
            "mem_total_comprimida": mem_comp,
            "reducao_pct": round(reducao, 2),
            "ratio_x": round(mem_original / max(1, mem_comp), 2),
            "tempo_ms": round((t_k + t_v) * 1000, 2)
        }

        print(f" {reducao:.1f}% redução ({mem_comp / 1024:.1f}KB) MSE={mse_k:.4f}/{mse_v:.4f} em {(t_k+t_v)*1000:.0f}ms")

    res["experimentos"]["compressao"] = {
        "config": {
            "seq_len": seq_len, "n_heads": n_heads, "head_dim": head_dim,
            "mem_original_bytes": mem_original
        },
        "por_k": resultados_k
    }

    # Extrapolação para modelos reais
    modelos = {
        "GPT2-small": {"seq": 1024, "heads": 12, "dim": 64},
        "LLaMA-7B": {"seq": 4096, "heads": 32, "dim": 128},
        "LLaMA-70B": {"seq": 8192, "heads": 64, "dim": 128},
    }
    print("\n  ▶ Extrapolação para modelos reais (K=256):")
    extrapol = {}
    for nome, cfg in modelos.items():
        mem_orig = cfg["seq"] * cfg["heads"] * cfg["dim"] * 4 * 2  # K+V, float32
        # Indices: int16 * seq * heads * 2 + Codebook: float32 * 256 * dim * 2
        mem_comp = cfg["seq"] * cfg["heads"] * 2 * 2 + 256 * cfg["dim"] * 4 * 2
        ratio = mem_orig / max(1, mem_comp)
        red = (1 - mem_comp / mem_orig) * 100
        extrapol[nome] = {
            "mem_original_MB": round(mem_orig / 1e6, 1),
            "mem_comprimida_MB": round(mem_comp / 1e6, 1),
            "reducao_pct": round(red, 1),
            "ratio_x": round(ratio, 1)
        }
        print(f"    {nome:>12}: {mem_orig/1e6:.1f}MB → {mem_comp/1e6:.1f}MB ({red:.0f}% redução, {ratio:.1f}x)")

    res["experimentos"]["extrapolacao"] = extrapol

    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    out = os.path.join(RESULTADOS_DIR, 'lab06_results.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Salvo em: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
