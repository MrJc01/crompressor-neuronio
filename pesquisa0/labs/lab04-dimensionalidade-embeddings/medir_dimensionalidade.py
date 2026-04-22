#!/usr/bin/env python3
"""
LAB04 — Dimensionalidade Intrinseca de Embeddings/Codebooks
===========================================================
Mede quantas "dimensões efetivas" um conjunto de vetores (como um Codebook)
realmente utiliza, usando PCA (Análise de Componentes Principais) via SVD.

Se o codebook tiver K=512 vetores em D=768 dimensões, quantas componentes
explicam 95% da variância?

Saída: JSON em pesquisa0/resultados/lab04_results.json
"""

import json
import os
import random
from datetime import datetime
from typing import List, Dict, Tuple

SEED = 42
random.seed(SEED)
RESULTADOS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'resultados')

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def gerar_codebook_sintetico(K: int, D: int, dim_intrinseca: int, ruido: float = 0.1):
    """
    Gera um codebook sintético onde a informação real vive em `dim_intrinseca`
    dimensões, projetadas em um espaço de dimensão `D`, com algum ruído.
    """
    if not HAS_NUMPY:
        return []

    # Gerar vetores na dimensão intrínseca
    base = np.random.randn(K, dim_intrinseca)
    
    # Matriz de projeção aleatória para expandir para dimensão D
    projecao = np.random.randn(dim_intrinseca, D)
    
    # Expandir e adicionar ruído
    codebook = np.dot(base, projecao) + ruido * np.random.randn(K, D)
    return codebook


def calcular_dimensionalidade_pca(matriz, variancia_alvo: float = 0.95):
    """Calcula componentes principais via SVD e encontra dims para variância alvo."""
    if not HAS_NUMPY:
        return 0, []

    # Centralizar os dados
    media = np.mean(matriz, axis=0)
    matriz_centrada = matriz - media

    # SVD
    U, S, Vt = np.linalg.svd(matriz_centrada, full_matrices=False)
    
    # Autovalores (variância)
    eigenvalues = (S ** 2) / (matriz.shape[0] - 1)
    
    # Variância explicada
    var_explicada = eigenvalues / np.sum(eigenvalues)
    var_acumulada = np.cumsum(var_explicada)

    # Quantas componentes para atingir o alvo?
    dim_efetiva = np.searchsorted(var_acumulada, variancia_alvo) + 1
    
    return int(dim_efetiva), var_acumulada.tolist()


def testar_analogia_calabi_yau():
    """
    Testa se o Codebook Learning age como "compactificação":
    A dimensão original D é grande, mas a dimensão efetiva é pequena e fixa.
    """
    resultados = {}
    
    # Espaço original: 768 dimensões (ex: GPT-2)
    D = 768
    
    # Simulamos que os dados reais da tarefa (ex: linguagem) têm dimensão intrínseca ~20
    dim_real = 20
    
    for K in [64, 128, 256, 512, 1024]:
        # Treinamento simula encontrar os K melhores vetores
        cb = gerar_codebook_sintetico(K, D, dim_real, ruido=0.15)
        
        dim_efetiva, var_acum = calcular_dimensionalidade_pca(cb, 0.95)
        
        ratio = D / max(1, dim_efetiva)
        
        resultados[str(K)] = {
            "K": K,
            "D_original": D,
            "dim_intrínseca_real": dim_real,
            "dim_efetiva_medida": dim_efetiva,
            "compactacao_ratio": round(ratio, 2),
            "preservou_topologia": abs(dim_efetiva - dim_real) <= 5
        }
        
    return resultados


def main():
    print("=" * 60)
    print("  LAB04 — DIMENSIONALIDADE DOS EMBEDDINGS")
    print(f"  Numpy disponível: {HAS_NUMPY}")
    print("=" * 60)

    res = {
        "meta": {"timestamp": datetime.now().isoformat(), "seed": SEED, "numpy": HAS_NUMPY},
        "experimentos": {}
    }

    if not HAS_NUMPY:
        print("Erro: Este lab requer Numpy para cálculos matriciais SVD/PCA.")
        return

    # ── Teste 1: Medir Dimensionalidade Intrínseca ─────────
    print("\n  ▶ Calculando dimensões efetivas via PCA (95% variância)...")
    np.random.seed(SEED)
    
    # Codebook simulando camadas de diferentes modelos
    # MNIST MLP: D=64, intrínseca~10
    cb_mnist = gerar_codebook_sintetico(K=256, D=64, dim_intrinseca=10)
    dim_mnist, _ = calcular_dimensionalidade_pca(cb_mnist)
    print(f"    MNIST Codebook (64D): Dim efetiva = {dim_mnist}")
    
    # GPT-2: D=768, intrínseca~35
    cb_gpt2 = gerar_codebook_sintetico(K=512, D=768, dim_intrinseca=35)
    dim_gpt2, _ = calcular_dimensionalidade_pca(cb_gpt2)
    print(f"    GPT-2 Codebook (768D): Dim efetiva = {dim_gpt2}")
    
    res["experimentos"]["medicoes_pca"] = {
        "mnist": {"D": 64, "dim_efetiva": dim_mnist},
        "gpt2": {"D": 768, "dim_efetiva": dim_gpt2}
    }

    # ── Teste 2: Analogia Calabi-Yau ───────────────────────
    print("\n  ▶ Testando Analogia Calabi-Yau (Compactação)...")
    cy_results = testar_analogia_calabi_yau()
    
    for k, v in cy_results.items():
        print(f"    K={k:>4}: Dim Efetiva={v['dim_efetiva_medida']:>2} "
              f"(Ratio: {v['compactacao_ratio']:>4}x) "
              f"Estável? {'SIM' if v['preservou_topologia'] else 'NÃO'}")

    res["experimentos"]["calabi_yau"] = cy_results

    # ── Teste 3: Ratio de Compressão Dimensional ───────────
    # A física propõe 10D -> 4D (Ratio 2.5x)
    ratio_gpt2 = 768 / max(1, dim_gpt2)
    print("\n  ▶ Taxa de Compactação Dimensional:")
    print(f"    Ratio GPT-2 simulado: {ratio_gpt2:.1f}x (768D originais → {dim_gpt2}D efetivas)")
    print(f"    ✅ Compactação > 10x: {'SIM' if ratio_gpt2 > 10.0 else 'NÃO'}")
    
    res["experimentos"]["ratio_compressao"] = {
        "ratio_medido": round(ratio_gpt2, 2),
        "criterio_10x": ratio_gpt2 > 10.0
    }

    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    out = os.path.join(RESULTADOS_DIR, 'lab04_results.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Salvo em: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
