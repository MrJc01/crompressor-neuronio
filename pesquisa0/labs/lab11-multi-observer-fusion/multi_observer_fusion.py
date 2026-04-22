#!/usr/bin/env python3
"""
LAB11 — Multi Observer Fusion
==============================
Extensão do Lab02: fusão de observadores com pesos adaptativos
baseados em confiança + "observador virtual" por interpolação.

Saída: JSON em pesquisa0/resultados/lab11_results.json
"""

import json
import os
import math
import random
import time
from datetime import datetime
from typing import List, Tuple, Dict

SEED = 42
random.seed(SEED)
RESULTADOS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'resultados')


# ─── Sinal Ground Truth ──────────────────────────────────────
def gerar_sinal(duracao_s=10.0, taxa=10000):
    """Sinal complexo com pulsos."""
    n = int(duracao_s * taxa)
    timestamps = [i / taxa for i in range(n)]
    sinal = []
    for t in timestamps:
        s = math.sin(2 * math.pi * 1.0 * t)
        s += 0.5 * math.sin(2 * math.pi * 5.0 * t)
        s += 0.3 * math.sin(2 * math.pi * 20.0 * t)
        # Pulsos em t=2.0, 5.5, 8.3
        if 2.0 < t < 2.01 or 5.5 < t < 5.51 or 8.3 < t < 8.31:
            s += 3.0
        sinal.append(s)
    return timestamps, sinal


def amostrar(timestamps, sinal, taxa_hz, ruido=0.05, corrompido_pct=0.0):
    """Amostra com ruído e possível corrupção."""
    intervalo = 1.0 / taxa_hz
    t_obs, s_obs = [], []
    t = 0.0
    while t < timestamps[-1]:
        idx = int(t * 10000)
        if 0 <= idx < len(sinal):
            val = sinal[idx] + random.gauss(0, ruido)
            # Corrupção aleatória
            if random.random() < corrompido_pct:
                val = random.gauss(0, 5)  # Valor totalmente errado
            t_obs.append(t)
            s_obs.append(val)
        t += intervalo
    return t_obs, s_obs


# ─── Merge Adaptativo (com pesos) ────────────────────────────
def calcular_confianca(t_obs, s_obs, t_real, s_real, janela=0.05):
    """Calcula confiança de um observador comparando com ground truth em janela."""
    erros = []
    for t, s in zip(t_obs, s_obs):
        idx = min(int(t * 10000), len(s_real) - 1)
        erros.append((s - s_real[idx]) ** 2)
    if not erros:
        return 0.0
    mse = sum(erros) / len(erros)
    # Confiança: inverso do MSE (normalizado)
    return 1.0 / (1.0 + mse)


def merge_ponderado(observadores, pesos, resolucao=0.0001):
    """Merge com pesos baseados em confiança."""
    todos_t = {}
    for (t_obs, s_obs), peso in zip(observadores, pesos):
        for t, s in zip(t_obs, s_obs):
            t_key = round(t / resolucao) * resolucao
            if t_key not in todos_t:
                todos_t[t_key] = {"soma": 0.0, "peso_total": 0.0}
            todos_t[t_key]["soma"] += s * peso
            todos_t[t_key]["peso_total"] += peso

    t_merge = sorted(todos_t.keys())
    s_merge = [todos_t[t]["soma"] / max(0.0001, todos_t[t]["peso_total"]) for t in t_merge]
    return t_merge, s_merge


def merge_simples(observadores, resolucao=0.0001):
    """Merge com média simples (sem pesos)."""
    todos_t = {}
    for t_obs, s_obs in observadores:
        for t, s in zip(t_obs, s_obs):
            t_key = round(t / resolucao) * resolucao
            if t_key not in todos_t:
                todos_t[t_key] = []
            todos_t[t_key].append(s)
    t_merge = sorted(todos_t.keys())
    s_merge = [sum(v) / len(v) for v in [todos_t[t] for t in t_merge]]
    return t_merge, s_merge


# ─── Observador Virtual ──────────────────────────────────────
def criar_observador_virtual(obs_a, obs_b, pos_virtual=0.5):
    """Cria observador virtual interpolando entre A e B."""
    t_a, s_a = obs_a
    t_b, s_b = obs_b
    # Usar timestamps de A como base
    t_v, s_v = [], []
    idx_b = 0
    for i, t in enumerate(t_a):
        # Encontrar timestamp mais próximo em B
        while idx_b < len(t_b) - 1 and t_b[idx_b + 1] <= t:
            idx_b += 1
        if idx_b < len(s_b):
            # Interpolação linear baseada em posição virtual
            val = s_a[i] * (1 - pos_virtual) + s_b[idx_b] * pos_virtual
            t_v.append(t)
            s_v.append(val)
    return t_v, s_v


# ─── SNR ──────────────────────────────────────────────────────
def calcular_snr(s_obs, s_real, t_obs, taxa_real=10000):
    """SNR em dB."""
    erros = []
    for t, s in zip(t_obs, s_obs):
        idx = min(int(t * taxa_real), len(s_real) - 1)
        erros.append((s - s_real[idx]) ** 2)
    if not erros:
        return 0.0
    p_sinal = sum(s ** 2 for s in s_obs) / len(s_obs)
    p_ruido = sum(erros) / len(erros)
    if p_ruido < 1e-10:
        return 100.0
    return 10 * math.log10(p_sinal / p_ruido)


def main():
    print("=" * 60)
    print("  LAB11 — MULTI OBSERVER FUSION")
    print("=" * 60)

    res = {"meta": {"timestamp": datetime.now().isoformat(), "seed": SEED},
           "experimentos": {}}

    random.seed(SEED)
    t_real, s_real = gerar_sinal()

    # ── Teste 1: Merge Ponderado vs Simples ───────────────
    print("\n  ▶ Merge Ponderado vs Simples (com observador corrompido)...")
    obs_bom1 = amostrar(t_real, s_real, 100, ruido=0.05)
    obs_bom2 = amostrar(t_real, s_real, 500, ruido=0.03)
    obs_ruim = amostrar(t_real, s_real, 100, ruido=0.1, corrompido_pct=0.2)

    observadores = [obs_bom1, obs_bom2, obs_ruim]

    # Confiança
    conf = [calcular_confianca(t, s, t_real, s_real) for t, s in observadores]
    print(f"    Confiança: bom1={conf[0]:.3f} bom2={conf[1]:.3f} ruim={conf[2]:.3f}")

    # Merge simples
    t_ms, s_ms = merge_simples(observadores)
    snr_simples = calcular_snr(s_ms, s_real, t_ms)

    # Merge ponderado
    t_mp, s_mp = merge_ponderado(observadores, conf)
    snr_ponderado = calcular_snr(s_mp, s_real, t_mp)

    # Baseline: só o melhor observador
    snr_melhor = calcular_snr(obs_bom2[1], s_real, obs_bom2[0])

    print(f"    SNR Melhor individual: {snr_melhor:.2f} dB")
    print(f"    SNR Merge Simples:     {snr_simples:.2f} dB")
    print(f"    SNR Merge Ponderado:   {snr_ponderado:.2f} dB")
    print(f"    ✅ Ponderado > Simples: {'SIM' if snr_ponderado > snr_simples else 'NÃO'}")

    res["experimentos"]["merge_ponderado_vs_simples"] = {
        "confiancas": [round(c, 4) for c in conf],
        "snr_melhor_individual_db": round(snr_melhor, 2),
        "snr_merge_simples_db": round(snr_simples, 2),
        "snr_merge_ponderado_db": round(snr_ponderado, 2),
        "ponderado_melhor": snr_ponderado > snr_simples,
    }

    # ── Teste 2: Observador Virtual ───────────────────────
    print("\n  ▶ Observador Virtual (interpolação entre A e B)...")
    obs_a = amostrar(t_real, s_real, 200, ruido=0.04)
    obs_b = amostrar(t_real, s_real, 200, ruido=0.04)

    # Criar virtual como mix 50/50
    t_v, s_v = criar_observador_virtual(obs_a, obs_b, pos_virtual=0.5)
    snr_a = calcular_snr(obs_a[1], s_real, obs_a[0])
    snr_b = calcular_snr(obs_b[1], s_real, obs_b[0])
    snr_v = calcular_snr(s_v, s_real, t_v)

    print(f"    SNR Obs A: {snr_a:.2f} dB")
    print(f"    SNR Obs B: {snr_b:.2f} dB")
    print(f"    SNR Virtual: {snr_v:.2f} dB")
    print(f"    ✅ Virtual viável: {'SIM' if snr_v > 0 else 'NÃO'}")

    res["experimentos"]["observador_virtual"] = {
        "snr_a_db": round(snr_a, 2),
        "snr_b_db": round(snr_b, 2),
        "snr_virtual_db": round(snr_v, 2),
        "virtual_viavel": snr_v > 0,
    }

    # ── Teste 3: Escalabilidade ───────────────────────────
    print("\n  ▶ Escalabilidade (1 a 50 observadores ponderados)...")
    escala = {}
    for n_obs in [1, 3, 5, 10, 20, 50]:
        random.seed(SEED)
        obs_list = []
        for i in range(n_obs):
            taxa = random.choice([50, 100, 200, 500])
            ruido = random.uniform(0.02, 0.15)
            corr = 0.1 if random.random() < 0.2 else 0.0  # 20% chance de corrompido
            obs_list.append(amostrar(t_real, s_real, taxa, ruido, corr))

        confs = [calcular_confianca(t, s, t_real, s_real) for t, s in obs_list]
        t_m, s_m = merge_ponderado(obs_list, confs)
        snr = calcular_snr(s_m, s_real, t_m)

        escala[str(n_obs)] = {"n_obs": n_obs, "snr_db": round(snr, 2)}
        print(f"    {n_obs:>3d} obs: SNR={snr:.2f} dB")

    res["experimentos"]["escalabilidade"] = escala

    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    out = os.path.join(RESULTADOS_DIR, 'lab11_results.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Salvo em: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
