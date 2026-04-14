"""
Tensor-Vivo — Exp0 Parte 3: CDC com parâmetros refinados

O primeiro teste com avg=512 deu 0% dedup. Isso é esperado:
pesos float32 são semi-aleatórios byte a byte, então chunks grandes
de bytes raw quase nunca colidem.

Neste script exploramos 3 estratégias:
  A) CDC com chunks muito menores (avg=64)
  B) Quantização dos pesos ANTES do CDC (float32 → int8, reduz entropia)
  C) CDC sobre os pesos como vetores (cada row/neurônio = 1 unidade)
"""

import os
import sys
import json
import math
import hashlib
import numpy as np
import torch
from collections import Counter

from fastcdc.fastcdc_py import fastcdc_py

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "dados")
MODEL_PATH = os.path.join(DATA_DIR, "mnist_mlp.pt")
RESULTS_PATH = os.path.join(DATA_DIR, "exp0_refined_results.json")

sys.path.insert(0, SCRIPT_DIR)
from train_mnist import SimpleMLP


def shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    freq = Counter(data)
    n = len(data)
    return -sum((c/n) * math.log2(c/n) for c in freq.values())


def run_cdc(raw_bytes, label, min_s=64, avg_s=512, max_s=4096):
    tmp = os.path.join(DATA_DIR, f"_tmp_{label}.bin")
    with open(tmp, "wb") as f:
        f.write(raw_bytes)
    chunks = list(fastcdc_py(tmp, min_size=min_s, avg_size=avg_s, max_size=max_s))
    os.remove(tmp)

    hashes = []
    for c in chunks:
        d = raw_bytes[c.offset:c.offset + c.length]
        hashes.append(hashlib.sha256(d).hexdigest())

    total = len(hashes)
    unique = len(set(hashes))
    return {
        "label": label,
        "chunks": total,
        "unique": unique,
        "dedup": total - unique,
        "dedup_pct": round((total - unique) / total * 100, 2) if total else 0,
        "bytes": len(raw_bytes),
    }


def quantize_weights(tensor, bits=8):
    """Quantiza tensor float para intN reduzindo entropia."""
    t = tensor.detach().cpu().numpy().flatten()
    vmin, vmax = t.min(), t.max()
    scale = (2**bits - 1) / (vmax - vmin + 1e-10)
    quantized = np.clip(((t - vmin) * scale).round(), 0, 2**bits - 1)
    if bits <= 8:
        return quantized.astype(np.uint8).tobytes()
    else:
        return quantized.astype(np.uint16).tobytes()


def main():
    print("=" * 60)
    print("  TENSOR-VIVO — Exp0: CDC REFINADO")
    print("=" * 60)

    model = SimpleMLP()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

    results = {"strategies": []}

    # ═══════════════════════════════════════════════
    # ESTRATÉGIA A: CDC com chunks menores
    # ═══════════════════════════════════════════════
    # FastCDC limits: min>=64, avg>=256, max>=1024
    print("\n━━━ ESTRATÉGIA A: CDC com chunks variados ━━━")
    for name, param in model.named_parameters():
        if "weight" not in name:
            continue
        raw = param.detach().cpu().numpy().astype(np.float32).tobytes()
        for avg in [256, 512, 1024, 2048]:
            min_s = max(64, avg // 4)
            max_s = max(1024, avg * 4)
            r = run_cdc(raw, f"A_{name}_avg{avg}", min_s, avg, max_s)
            print(f"  {name} avg={avg}: {r['chunks']} chunks, {r['unique']} únicos, {r['dedup_pct']}% dedup")
            r["strategy"] = "A_varied_chunks"
            r["avg_size"] = avg
            r["layer"] = name
            results["strategies"].append(r)

    # ═══════════════════════════════════════════════
    # ESTRATÉGIA B: Quantizar ANTES do CDC
    # ═══════════════════════════════════════════════
    print("\n━━━ ESTRATÉGIA B: Quantizar pesos antes do CDC ━━━")
    for bits in [8, 4]:
        print(f"\n  Quantização para {bits}-bit:")
        all_quant = b""
        for name, param in model.named_parameters():
            if "weight" not in name:
                continue
            if bits == 4:
                # Pack 2 nibbles per byte
                t = param.detach().cpu().numpy().flatten()
                vmin, vmax = t.min(), t.max()
                scale = 15.0 / (vmax - vmin + 1e-10)
                q4 = np.clip(((t - vmin) * scale).round(), 0, 15).astype(np.uint8)
                # Pack pairs
                if len(q4) % 2 != 0:
                    q4 = np.append(q4, 0)
                packed = ((q4[0::2] << 4) | q4[1::2]).tobytes()
                quant_bytes = packed
            else:
                quant_bytes = quantize_weights(param, bits)
            all_quant += quant_bytes

            r = run_cdc(quant_bytes, f"B_{name}_{bits}bit", 64, 256, 1024)
            print(f"    {name}: {r['chunks']} chunks, {r['unique']} únicos, {r['dedup_pct']}% dedup")
            print(f"      Entropia: {shannon_entropy(quant_bytes):.4f} bits/byte (original float32: ~6.8)")
            r["strategy"] = f"B_quantize_{bits}bit"
            r["bits"] = bits
            r["layer"] = name
            results["strategies"].append(r)

        # CDC sobre blob inteiro quantizado
        r = run_cdc(all_quant, f"B_global_{bits}bit", 64, 256, 1024)
        print(f"  GLOBAL {bits}-bit: {r['chunks']} chunks, {r['unique']} únicos, {r['dedup_pct']}% dedup")
        r["strategy"] = f"B_quantize_{bits}bit_global"
        r["bits"] = bits
        results["strategies"].append(r)

    # ═══════════════════════════════════════════════
    # ESTRATÉGIA C: Cada neurônio (row) como unidade
    # ═══════════════════════════════════════════════
    print("\n━━━ ESTRATÉGIA C: Hash por NEURÔNIO (cada row = 1 unidade) ━━━")
    for name, param in model.named_parameters():
        if "weight" not in name:
            continue
        w = param.detach().cpu().numpy()
        rows = w.shape[0]

        # Hash de cada row como float32
        row_hashes_f32 = []
        for i in range(rows):
            h = hashlib.sha256(w[i].tobytes()).hexdigest()
            row_hashes_f32.append(h)

        unique_f32 = len(set(row_hashes_f32))
        dedup_f32 = rows - unique_f32

        # Hash de cada row como int8 quantizado
        row_hashes_q8 = []
        vmin, vmax = w.min(), w.max()
        scale = 255.0 / (vmax - vmin + 1e-10)
        w_q8 = np.clip(((w - vmin) * scale).round(), 0, 255).astype(np.uint8)

        for i in range(rows):
            h = hashlib.sha256(w_q8[i].tobytes()).hexdigest()
            row_hashes_q8.append(h)

        unique_q8 = len(set(row_hashes_q8))
        dedup_q8 = rows - unique_q8

        # Similaridade: quantos pares de neurônios têm cosine > 0.99?
        from numpy.linalg import norm
        similar_count = 0
        very_similar = 0
        norms = np.array([norm(w[i]) for i in range(rows)])
        for i in range(rows):
            for j in range(i+1, rows):
                if norms[i] < 1e-6 or norms[j] < 1e-6:
                    continue
                cos = np.dot(w[i], w[j]) / (norms[i] * norms[j])
                if cos > 0.95:
                    similar_count += 1
                if cos > 0.99:
                    very_similar += 1

        total_pairs = rows * (rows - 1) // 2

        print(f"  {name} ({rows} neurônios, {w.shape[1]} dims):")
        print(f"    Float32 hash: {unique_f32} únicos, {dedup_f32} duplicados ({dedup_f32/rows*100:.1f}%)")
        print(f"    Int8 hash:    {unique_q8} únicos, {dedup_q8} duplicados ({dedup_q8/rows*100:.1f}%)")
        print(f"    Cos > 0.95:   {similar_count}/{total_pairs} pares ({similar_count/max(1,total_pairs)*100:.2f}%)")
        print(f"    Cos > 0.99:   {very_similar}/{total_pairs} pares ({very_similar/max(1,total_pairs)*100:.2f}%)")

        results["strategies"].append({
            "strategy": "C_per_neuron",
            "layer": name,
            "neurons": rows,
            "dims": w.shape[1],
            "unique_f32": unique_f32,
            "dedup_f32": dedup_f32,
            "unique_q8": unique_q8,
            "dedup_q8": dedup_q8,
            "similar_095": similar_count,
            "similar_099": very_similar,
            "total_pairs": total_pairs,
        })

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Resultados salvos em: {RESULTS_PATH}")

    # ═══════════════════════════════════════════════
    # CONCLUSÃO
    # ═══════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  CONCLUSÃO REFINADA")
    print(f"{'='*60}")
    print("""
  O CDC clássico (hash exato de bytes) tem dificuldade com pesos float32
  porque cada peso é numericamente único — mesmo neurônios funcionalmente
  similares diferem em casas decimais.

  PORÉM, a análise de SIMILARIDADE (cosine > 0.95) revela se existem
  neurônios "quase iguais" que um codebook poderia representar com
  entradas compartilhadas.

  A estratégia real do Tensor-Vivo não é dedup por hash exato, mas sim:
  1. Agrupar neurônios similares em CLUSTERS (k-means)
  2. Representar cada cluster por 1 entrada de Codebook
  3. Cada neurônio aponta para sua entrada → weight sharing semântico

  Isso é o que a Fase 1 (Roundtrip com Codebook) vai testar.
""")


if __name__ == "__main__":
    main()
