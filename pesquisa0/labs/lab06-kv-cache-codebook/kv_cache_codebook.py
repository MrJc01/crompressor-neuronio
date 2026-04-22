#!/usr/bin/env python3
"""
LAB06 — Codebook Learning no KV Cache
=====================================
Aplica vetor quantização (Codebook) ao KV Cache de um Transformer simulado.
Em vez de guardar N tensores (seq_len, head_dim) de floats,
guardamos índices (int16) apontando para um dicionário de K vetores.

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


class KVCacheSimulator:
    """Simula o KV Cache de um Transformer (ex: LLaMA-like)."""
    def __init__(self, seq_len=4096, n_heads=32, head_dim=128):
        self.seq_len = seq_len
        self.n_heads = n_heads
        self.head_dim = head_dim
        
        # Simulando um KV Cache cheio: (seq_len, n_heads, head_dim)
        if HAS_NUMPY:
            np.random.seed(SEED)
            self.k_cache = np.random.randn(seq_len, n_heads, head_dim).astype(np.float32)
            self.v_cache = np.random.randn(seq_len, n_heads, head_dim).astype(np.float32)
            
            # Adicionando estrutura para não ser só ruído (para quantização funcionar melhor)
            # Padrões comuns ao longo do tempo
            base_pattern = np.random.randn(1, n_heads, head_dim)
            self.k_cache += base_pattern * 0.5
        
    def mem_size_bytes(self):
        if not HAS_NUMPY: return 0
        return self.k_cache.nbytes + self.v_cache.nbytes


def quantizar_cache(cache: 'np.ndarray', K: int = 512):
    """
    Aplica K-Means simplificado (Vector Quantization) para criar um codebook.
    Retorna o codebook e a matriz de índices.
    """
    seq_len, n_heads, head_dim = cache.shape
    
    # Flatten para (seq_len * n_heads, head_dim)
    flat_cache = cache.reshape(-1, head_dim)
    n_vectors = flat_cache.shape[0]
    
    # Inicializar codebook com amostras aleatórias do próprio cache
    indices_iniciais = np.random.choice(n_vectors, K, replace=False)
    codebook = flat_cache[indices_iniciais].copy()
    
    # 1 iteração de "K-Means" (apenas para simular o processo)
    # Distância Euclidiana (l2)
    # Para performance na simulação, não vamos fazer loop pesado.
    # Apenas calculamos distâncias por blocos.
    
    # Simplificação extrema para o lab rodar rápido:
    # Atribuir cada vetor ao mais próximo
    # Isso requer calcular (N, K) distâncias, que pode ser pesado.
    # N = 4096 * 32 = 131,072. K = 512. Matriz de dist: 131072 x 512.
    
    # Para não travar, fazemos um sub-sample para o codebook e depois
    # aproximação rápida.
    t0 = time.perf_counter()
    
    indices = np.zeros(n_vectors, dtype=np.int16)
    
    # Batch processing para não explodir a RAM
    batch_size = 16384
    for i in range(0, n_vectors, batch_size):
        end = min(i + batch_size, n_vectors)
        batch = flat_cache[i:end]
        
        # dists: (batch_size, K)
        # (A - B)^2 = A^2 + B^2 - 2AB
        A2 = np.sum(batch**2, axis=1, keepdims=True)
        B2 = np.sum(codebook**2, axis=1)
        AB = np.dot(batch, codebook.T)
        dists = A2 + B2 - 2*AB
        
        indices[i:end] = np.argmin(dists, axis=1)
        
    # Reconstrução
    reconstructed_flat = codebook[indices]
    reconstructed = reconstructed_flat.reshape(seq_len, n_heads, head_dim)
    
    t_quant = time.perf_counter() - t0
    
    # Calcular Erro (MSE)
    mse = np.mean((cache - reconstructed)**2)
    
    return indices.reshape(seq_len, n_heads), codebook, mse, t_quant


def main():
    print("=" * 60)
    print("  LAB06 — CODEBOOK NO KV CACHE")
    print(f"  Numpy disponível: {HAS_NUMPY}")
    print("=" * 60)

    res = {
        "meta": {"timestamp": datetime.now().isoformat(), "seed": SEED, "numpy": HAS_NUMPY},
        "experimentos": {}
    }

    if not HAS_NUMPY:
        print("Erro: Lab requer Numpy.")
        return

    print("\n  ▶ Comprimindo KV Cache de 2048 tokens...")
    seq_len = 2048
    
    simulator = KVCacheSimulator(seq_len=seq_len, n_heads=32, head_dim=128)
    mem_original = simulator.mem_size_bytes()
    
    resultados_k = {}
    
    for K in [128, 256, 512]:
        # Quantizar K Cache
        idx_k, cb_k, mse_k, t_k = quantizar_cache(simulator.k_cache, K)
        
        # Quantizar V Cache
        idx_v, cb_v, mse_v, t_v = quantizar_cache(simulator.v_cache, K)
        
        # Tamanho comprimido
        # Indices: int16 (2 bytes) * seq_len * n_heads * 2 (K e V)
        mem_indices = idx_k.nbytes + idx_v.nbytes
        # Codebook: float32 (4 bytes) * K * head_dim * 2 (K e V)
        mem_codebook = cb_k.nbytes + cb_v.nbytes
        
        mem_comprimido = mem_indices + mem_codebook
        reducao = (1 - mem_comprimido / mem_original) * 100
        
        resultados_k[str(K)] = {
            "K": K,
            "mse_k": float(mse_k),
            "mse_v": float(mse_v),
            "mse_medio": float((mse_k + mse_v) / 2),
            "memoria_indices_bytes": mem_indices,
            "memoria_codebook_bytes": mem_codebook,
            "memoria_total_comprimida": mem_comprimido,
            "reducao_pct": round(reducao, 2),
            "ratio_compressao_x": round(mem_original / max(1, mem_comprimido), 2),
            "tempo_quantizacao_ms": round((t_k + t_v) * 1000, 2)
        }
        
    exp = {
        "config": {
            "seq_len": seq_len,
            "n_heads": 32,
            "head_dim": 128,
            "memoria_original_bytes": mem_original
        },
        "por_k": resultados_k
    }
    
    mem_orig_mb = exp["config"]["memoria_original_bytes"] / (1024*1024)
    print(f"    Memória Original: {mem_orig_mb:.2f} MB")
    
    for K, dados in exp["por_k"].items():
        comp_mb = dados['memoria_total_comprimida'] / (1024*1024)
        print(f"    K={K:>4}: {comp_mb:>5.2f} MB ({dados['reducao_pct']:>5.1f}% redução, {dados['ratio_compressao_x']:>4.1f}x) "
              f"| Erro (MSE): {dados['mse_medio']:.4f}")
              
    res["experimentos"]["compressao"] = exp

    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    out = os.path.join(RESULTADOS_DIR, 'lab06_results.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Salvo em: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
