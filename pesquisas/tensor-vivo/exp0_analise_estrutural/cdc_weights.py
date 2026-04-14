"""
Tensor-Vivo — Experimento 0, Parte 2: Análise CDC dos Pesos

Carrega o MLP treinado, extrai os pesos como bytes raw,
roda FastCDC e analisa:
  - Quantidade de chunks total vs únicos
  - Taxa de deduplicação
  - Entropia de Shannon por chunk
  - Comparação entre camadas
"""

import os
import sys
import json
import math
import hashlib
import struct
import numpy as np
import torch
from collections import Counter

# FastCDC
from fastcdc.fastcdc_py import fastcdc_py

# ── Paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "dados")
MODEL_PATH = os.path.join(DATA_DIR, "mnist_mlp.pt")
RESULTS_PATH = os.path.join(DATA_DIR, "exp0_results.json")

# Precisamos importar o modelo
sys.path.insert(0, SCRIPT_DIR)
from train_mnist import SimpleMLP


def shannon_entropy(data: bytes) -> float:
    """Entropia de Shannon em bits/byte."""
    if not data:
        return 0.0
    freq = Counter(data)
    n = len(data)
    entropy = 0.0
    for count in freq.values():
        p = count / n
        entropy -= p * math.log2(p)
    return entropy


def run_cdc_on_bytes(raw_bytes: bytes, label: str,
                     min_size=64, avg_size=512, max_size=4096):
    """Roda FastCDC sobre bytes raw e retorna análise."""
    # fastcdc_py espera source como filepath ou bytes
    # Vamos escrever temporariamente num arquivo
    tmp_path = os.path.join(DATA_DIR, f"_tmp_{label}.bin")
    with open(tmp_path, "wb") as f:
        f.write(raw_bytes)

    chunks = list(fastcdc_py(tmp_path, min_size=min_size,
                              avg_size=avg_size, max_size=max_size))
    os.remove(tmp_path)

    # Análise
    hashes = []
    entropies = []
    sizes = []

    for chunk in chunks:
        chunk_data = raw_bytes[chunk.offset:chunk.offset + chunk.length]
        h = hashlib.sha256(chunk_data).hexdigest()
        hashes.append(h)
        entropies.append(shannon_entropy(chunk_data))
        sizes.append(chunk.length)

    unique_hashes = set(hashes)
    total = len(hashes)
    unique = len(unique_hashes)
    dedup = total - unique
    dedup_rate = (dedup / total * 100) if total > 0 else 0

    return {
        "label": label,
        "total_bytes": len(raw_bytes),
        "total_chunks": total,
        "unique_chunks": unique,
        "dedup_chunks": dedup,
        "dedup_rate_pct": round(dedup_rate, 4),
        "avg_chunk_size": round(np.mean(sizes), 1) if sizes else 0,
        "min_chunk_size": min(sizes) if sizes else 0,
        "max_chunk_size": max(sizes) if sizes else 0,
        "entropy_mean": round(float(np.mean(entropies)), 4) if entropies else 0,
        "entropy_std": round(float(np.std(entropies)), 4) if entropies else 0,
        "entropy_min": round(float(np.min(entropies)), 4) if entropies else 0,
        "entropy_max": round(float(np.max(entropies)), 4) if entropies else 0,
        "chunk_sizes": sizes,
        "chunk_entropies": [round(e, 4) for e in entropies],
    }


def main():
    print("=" * 60)
    print("  TENSOR-VIVO — Exp0: Análise CDC dos Pesos Neurais")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Modelo não encontrado em {MODEL_PATH}")
        print("   Rode primeiro: python train_mnist.py")
        return

    # ── Carregar modelo ──
    model = SimpleMLP()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    print(f"✅ Modelo carregado: {MODEL_PATH}")

    # ── Extrair pesos como bytes ──
    all_weights_bytes = b""
    layer_data = {}

    for name, param in model.named_parameters():
        raw = param.detach().cpu().numpy().astype(np.float32).tobytes()
        layer_data[name] = raw
        all_weights_bytes += raw
        print(f"  📦 {name}: {param.shape} → {len(raw):,} bytes")

    print(f"\n  📊 Total de bytes (todos os pesos): {len(all_weights_bytes):,}")

    # ── CDC sobre TODOS os pesos concatenados ──
    print(f"\n{'─'*60}")
    print("  🔬 CDC sobre blob COMPLETO (todos os pesos concatenados)")
    print(f"{'─'*60}")

    global_result = run_cdc_on_bytes(all_weights_bytes, "global_all_weights")

    print(f"  Chunks Total:     {global_result['total_chunks']}")
    print(f"  Chunks Únicos:    {global_result['unique_chunks']}")
    print(f"  Chunks Duplicados:{global_result['dedup_chunks']}")
    print(f"  Taxa de Dedup:    {global_result['dedup_rate_pct']:.2f}%")
    print(f"  Chunk Size Médio: {global_result['avg_chunk_size']:.0f} bytes")
    print(f"  Entropia Média:   {global_result['entropy_mean']:.4f} bits/byte")
    print(f"  Entropia Std:     {global_result['entropy_std']:.4f}")

    # ── CDC por camada individual ──
    print(f"\n{'─'*60}")
    print("  🔬 CDC por CAMADA INDIVIDUAL")
    print(f"{'─'*60}")

    layer_results = []
    for name, raw in layer_data.items():
        if len(raw) < 128:  # Pular biases muito pequenos
            print(f"  ⏩ {name}: {len(raw)} bytes (muito pequeno para CDC)")
            layer_results.append({
                "label": name,
                "total_bytes": len(raw),
                "skipped": True,
                "reason": "tamanho insuficiente para CDC"
            })
            continue

        result = run_cdc_on_bytes(raw, name)
        layer_results.append(result)
        print(f"  {name}:")
        print(f"    Chunks: {result['total_chunks']} total, {result['unique_chunks']} únicos, {result['dedup_rate_pct']:.2f}% dedup")
        print(f"    Entropia: μ={result['entropy_mean']:.4f} σ={result['entropy_std']:.4f} [{result['entropy_min']:.4f}, {result['entropy_max']:.4f}]")

    # ── CDC CROSS-LAYER: verificar dedup ENTRE camadas ──
    print(f"\n{'─'*60}")
    print("  🔬 Análise CROSS-LAYER (dedup entre camadas diferentes)")
    print(f"{'─'*60}")

    # Coletar todos os hashes de chunks de todas as camadas
    all_layer_hashes = {}  # hash → lista de camadas onde aparece
    for name, raw in layer_data.items():
        if len(raw) < 128:
            continue
        tmp_path = os.path.join(DATA_DIR, f"_tmp_cross_{name}.bin")
        with open(tmp_path, "wb") as f:
            f.write(raw)
        chunks = list(fastcdc_py(tmp_path, min_size=64, avg_size=512, max_size=4096))
        os.remove(tmp_path)
        for chunk in chunks:
            chunk_data = raw[chunk.offset:chunk.offset + chunk.length]
            h = hashlib.sha256(chunk_data).hexdigest()
            if h not in all_layer_hashes:
                all_layer_hashes[h] = []
            all_layer_hashes[h].append(name)

    cross_layer_shared = {h: layers for h, layers in all_layer_hashes.items()
                          if len(set(layers)) > 1}
    total_unique_across = len(all_layer_hashes)
    shared_across = len(cross_layer_shared)

    print(f"  Hashes únicos (todas as camadas): {total_unique_across}")
    print(f"  Hashes compartilhados entre ≥2 camadas: {shared_across}")
    if shared_across > 0:
        print(f"  ✅ DEDUP CROSS-LAYER ENCONTRADA!")
        for h, layers in list(cross_layer_shared.items())[:5]:
            unique_layers = list(set(layers))
            print(f"    Hash {h[:16]}... → compartilhado por: {unique_layers}")
    else:
        print(f"  ℹ️ Nenhuma dedup cross-layer encontrada (normal para modelos pequenos)")

    # ── Salvar resultados ──
    results = {
        "experiment": "exp0_analise_estrutural",
        "model": "SimpleMLP_784_256_128_10",
        "cdc_params": {"min_size": 64, "avg_size": 512, "max_size": 4096},
        "global_analysis": {k: v for k, v in global_result.items()
                           if k not in ("chunk_sizes", "chunk_entropies")},
        "global_chunk_sizes": global_result["chunk_sizes"],
        "global_chunk_entropies": global_result["chunk_entropies"],
        "per_layer": [{k: v for k, v in r.items()
                       if k not in ("chunk_sizes", "chunk_entropies")}
                      for r in layer_results],
        "cross_layer": {
            "total_unique_hashes": total_unique_across,
            "shared_across_layers": shared_across,
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Resultados salvos em: {RESULTS_PATH}")

    # ── Veredicto ──
    print(f"\n{'='*60}")
    print("  VEREDICTO PRELIMINAR")
    print(f"{'='*60}")
    if global_result["dedup_rate_pct"] > 0:
        print(f"  ✅ CDC ENCONTROU DEDUP nos pesos: {global_result['dedup_rate_pct']:.2f}%")
        print(f"     Isso prova que EXISTE redundância estrutural nos pesos")
        print(f"     que o Content-Defined Chunking consegue detectar.")
        print(f"     → PROSSEGUIR COM FASE 1 (Roundtrip & Codebook)")
    else:
        print(f"  ⚠️ CDC NÃO encontrou dedup (0%) neste modelo.")
        print(f"     Possíveis razões:")
        print(f"     - Modelo muito pequeno (235K params)")
        print(f"     - CDC params (avg=512) podem ser grandes demais para um MLP MNIST")
        print(f"     → Testar com min=16, avg=64, max=512 (chunks menores)")
        print(f"     → Testar com modelo maior (CNN CIFAR ou Transformer)")

    print(f"\n  Entropia média dos chunks: {global_result['entropy_mean']:.4f} bits/byte")
    if global_result['entropy_mean'] < 7.5:
        print(f"  ✅ Entropia < 7.5 indica que os pesos NÃO são random noise.")
        print(f"     Existe estrutura comprimível.")
    else:
        print(f"  ℹ️ Entropia alta ({global_result['entropy_mean']:.2f}) — pesos se comportam")
        print(f"     como dados quase-aleatórios, típico de modelos treinados.")


if __name__ == "__main__":
    main()
