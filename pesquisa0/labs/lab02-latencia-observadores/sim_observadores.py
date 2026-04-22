#!/usr/bin/env python3
"""
LAB02 — Simulação de Multi-Observadores
=========================================
Simula N observadores com taxas de amostragem diferentes
observando o mesmo evento. Mede ganho de "Post-Sync Merge".

Saída: JSON em pesquisa0/resultados/lab02_results.json
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

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ─── Evento Sintético ────────────────────────────────────────
def gerar_evento(duracao_s: float = 10.0, taxa_base: int = 10000) -> Tuple[List[float], List[float]]:
    """Gera sinal complexo (onda composta + ruído + micro-eventos)."""
    n = int(duracao_s * taxa_base)
    timestamps = [i / taxa_base for i in range(n)]
    sinal = []
    for t in timestamps:
        # Onda base (soma de senoides)
        s = math.sin(2 * math.pi * 1.0 * t)         # 1 Hz
        s += 0.5 * math.sin(2 * math.pi * 5.0 * t)  # 5 Hz
        s += 0.3 * math.sin(2 * math.pi * 20.0 * t) # 20 Hz (alta freq)
        # Micro-eventos (pulsos curtos)
        if 2.0 < t < 2.01 or 5.5 < t < 5.51 or 8.3 < t < 8.31:
            s += 3.0  # Pulso forte
        # Ruído
        s += random.gauss(0, 0.1)
        sinal.append(s)
    return timestamps, sinal


# ─── Observador ───────────────────────────────────────────────
def amostrar(timestamps: List[float], sinal: List[float],
             taxa_hz: int, offset_s: float = 0.0,
             ruido_sensor: float = 0.05) -> Tuple[List[float], List[float]]:
    """Amostra o sinal a uma taxa específica com offset temporal."""
    intervalo = 1.0 / taxa_hz
    t_obs, s_obs = [], []
    t = offset_s
    while t < timestamps[-1]:
        # Encontrar índice mais próximo
        idx = int(t * 10000)
        if 0 <= idx < len(sinal):
            t_obs.append(t)
            s_obs.append(sinal[idx] + random.gauss(0, ruido_sensor))
        t += intervalo
    return t_obs, s_obs


# ─── Post-Sync Merge ─────────────────────────────────────────
def merge_observadores(observadores: List[Tuple[List[float], List[float]]],
                       resolucao: float = 0.0001) -> Tuple[List[float], List[float]]:
    """Combina dados de múltiplos observadores por alinhamento temporal."""
    # Coletar todos os timestamps
    todos_t = {}
    for t_obs, s_obs in observadores:
        for t, s in zip(t_obs, s_obs):
            t_key = round(t / resolucao) * resolucao
            if t_key not in todos_t:
                todos_t[t_key] = []
            todos_t[t_key].append(s)

    # Média dos valores no mesmo timestamp
    t_merge = sorted(todos_t.keys())
    s_merge = [sum(v) / len(v) for v in [todos_t[t] for t in t_merge]]
    return t_merge, s_merge


# ─── Métricas ─────────────────────────────────────────────────
def calcular_snr(sinal_obs: List[float], sinal_real: List[float],
                 t_obs: List[float], t_real: List[float]) -> float:
    """Calcula SNR (Signal-to-Noise Ratio) em dB."""
    # Interpolar sinal real para os timestamps do observador
    erros = []
    for t, s in zip(t_obs, sinal_obs):
        idx = min(int(t * 10000), len(sinal_real) - 1)
        erro = s - sinal_real[idx]
        erros.append(erro ** 2)

    if not erros:
        return 0.0

    # SNR = 10 * log10(P_sinal / P_ruido)
    p_sinal = sum(s ** 2 for s in sinal_obs) / len(sinal_obs)
    p_ruido = sum(erros) / len(erros)
    if p_ruido < 1e-10:
        return 100.0
    return 10 * math.log10(p_sinal / p_ruido)


def detectar_micro_eventos(t_obs: List[float], s_obs: List[float],
                           threshold: float = 2.5) -> List[float]:
    """Detecta pulsos (micro-eventos) acima do threshold."""
    eventos = []
    for t, s in zip(t_obs, s_obs):
        if abs(s) > threshold:
            eventos.append(round(t, 4))
    return eventos


# ─── Simulação Principal ─────────────────────────────────────
def run_simulacao(n_extra_obs: int = 0):
    """Roda simulação com observadores A, B, C + extras."""
    random.seed(SEED)
    t_real, s_real = gerar_evento(duracao_s=10.0)

    # Micro-eventos reais (ground truth)
    eventos_reais = [2.005, 5.505, 8.305]

    # Observadores fixos
    t_a, s_a = amostrar(t_real, s_real, taxa_hz=10, offset_s=0.0)     # Humano lento
    t_b, s_b = amostrar(t_real, s_real, taxa_hz=1000, offset_s=0.0)   # Máquina rápida
    t_c, s_c = amostrar(t_real, s_real, taxa_hz=10, offset_s=2.0)     # Distante

    observadores = [(t_a, s_a), (t_b, s_b), (t_c, s_c)]

    # Observadores extras
    for i in range(n_extra_obs):
        taxa = random.choice([10, 50, 100, 500])
        offset = random.uniform(0, 3)
        t_x, s_x = amostrar(t_real, s_real, taxa_hz=taxa, offset_s=offset)
        observadores.append((t_x, s_x))

    # Merge
    t_m, s_m = merge_observadores(observadores)

    # SNR
    snr_a = calcular_snr(s_a, s_real, t_a, t_real)
    snr_b = calcular_snr(s_b, s_real, t_b, t_real)
    snr_m = calcular_snr(s_m, s_real, t_m, t_real)

    # Detecção de micro-eventos
    ev_a = detectar_micro_eventos(t_a, s_a)
    ev_b = detectar_micro_eventos(t_b, s_b)
    ev_m = detectar_micro_eventos(t_m, s_m)

    # Cobertura: quantos dos 3 eventos reais foram detectados (±0.1s)
    def cobertura(detectados, reais, tolerancia=0.1):
        cobertos = 0
        for r in reais:
            for d in detectados:
                if abs(d - r) < tolerancia:
                    cobertos += 1
                    break
        return cobertos / len(reais) * 100

    cob_a = cobertura(ev_a, eventos_reais)
    cob_b = cobertura(ev_b, eventos_reais)
    cob_m = cobertura(ev_m, eventos_reais)

    return {
        "n_observadores": 3 + n_extra_obs,
        "amostras_A": len(s_a),
        "amostras_B": len(s_b),
        "amostras_merge": len(s_m),
        "snr_A_db": round(snr_a, 2),
        "snr_B_db": round(snr_b, 2),
        "snr_merge_db": round(snr_m, 2),
        "snr_merge_maior": snr_m > max(snr_a, snr_b),
        "eventos_detectados_A": len(ev_a),
        "eventos_detectados_B": len(ev_b),
        "eventos_detectados_merge": len(ev_m),
        "cobertura_A_pct": round(cob_a, 1),
        "cobertura_B_pct": round(cob_b, 1),
        "cobertura_merge_pct": round(cob_m, 1),
    }


def run_delta_merge():
    """Testa merge usando Deltas: B envia só o delta vs A."""
    random.seed(SEED)
    t_real, s_real = gerar_evento(duracao_s=10.0)
    t_a, s_a = amostrar(t_real, s_real, taxa_hz=10)
    t_b, s_b = amostrar(t_real, s_real, taxa_hz=1000)

    # Dados brutos de B
    bytes_brutos_b = len(s_b) * 8  # float64

    # Delta: B só envia amostras em timestamps que A NÃO tem
    t_a_set = set(round(t, 4) for t in t_a)
    delta_t, delta_s = [], []
    for t, s in zip(t_b, s_b):
        if round(t, 4) not in t_a_set:
            delta_t.append(t)
            delta_s.append(s)

    bytes_delta = len(delta_s) * 8
    reducao = (1 - bytes_delta / max(1, bytes_brutos_b)) * 100

    return {
        "bytes_brutos_B": bytes_brutos_b,
        "bytes_delta": bytes_delta,
        "reducao_pct": round(reducao, 1),
        "amostras_bruto": len(s_b),
        "amostras_delta": len(delta_s),
    }


def main():
    print("=" * 60)
    print("  LAB02 — MULTI-OBSERVADORES")
    print("=" * 60)

    res = {"meta": {"timestamp": datetime.now().isoformat(), "seed": SEED},
           "experimentos": {}}

    # ── 3 observadores ────────────────────────────────────
    print("\n  ▶ 3 observadores (A=10Hz, B=1000Hz, C=10Hz+offset)...")
    r3 = run_simulacao(0)
    print(f"    SNR:  A={r3['snr_A_db']}dB  B={r3['snr_B_db']}dB  Merge={r3['snr_merge_db']}dB")
    print(f"    ✅ Merge SNR > individual: {r3['snr_merge_maior']}")
    print(f"    Cobertura: A={r3['cobertura_A_pct']}%  B={r3['cobertura_B_pct']}%  Merge={r3['cobertura_merge_pct']}%")
    res["experimentos"]["3_observadores"] = r3

    # ── Curva de saturação ────────────────────────────────
    print("\n  ▶ Curva de saturação (1→50 observadores extras)...")
    saturacao = {}
    for n in [0, 2, 5, 10, 20, 47]:
        r = run_simulacao(n)
        total = r["n_observadores"]
        print(f"    {total:>3d} obs: SNR_merge={r['snr_merge_db']:>6.1f}dB  "
              f"Cob={r['cobertura_merge_pct']}%")
        saturacao[str(total)] = r
    res["experimentos"]["saturacao"] = saturacao

    # ── Delta merge ───────────────────────────────────────
    print("\n  ▶ Merge com Deltas (B envia só diferença vs A)...")
    dm = run_delta_merge()
    print(f"    Bruto B: {dm['bytes_brutos_B']:,} bytes ({dm['amostras_bruto']} amostras)")
    print(f"    Delta:   {dm['bytes_delta']:,} bytes ({dm['amostras_delta']} amostras)")
    print(f"    Redução: {dm['reducao_pct']}%")
    print(f"    ✅ >80% redução: {'SIM' if dm['reducao_pct'] > 80 else 'NÃO'}")
    res["experimentos"]["delta_merge"] = dm

    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    out = os.path.join(RESULTADOS_DIR, 'lab02_results.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Salvo em: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
