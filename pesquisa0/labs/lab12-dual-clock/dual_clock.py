#!/usr/bin/env python3
"""
LAB12 — Dual Clock (Teoria-F 12D)
===================================
Sistema com dois vetores temporais:
- Clock 1 (Inercial): avança com tempo real
- Clock 2 (Prospectivo): explora futuros possíveis

Saída: JSON em pesquisa0/resultados/lab12_results.json
"""

import json
import os
import time
import random
import math
from datetime import datetime
from typing import List, Dict

SEED = 42
random.seed(SEED)
RESULTADOS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'resultados')


class ClockInercial:
    """Clock 1: avança com dados reais, 1 step por vez."""
    def __init__(self):
        self.t = 0
        self.estado = 0.0       # posição real
        self.velocidade = 1.0

    def tick(self, ruido: float = 0.05):
        self.t += 1
        self.estado += self.velocidade + random.gauss(0, ruido)
        return self.estado


class ClockProspectivo:
    """Clock 2: explora N futuros a partir do estado atual."""
    def __init__(self, n_exploracoes: int = 100):
        self.n_exploracoes = n_exploracoes
        self.t_explorados = 0

    def explorar(self, estado_atual: float, velocidade: float,
                 n_branches: int = 5, profundidade: int = 20) -> Dict:
        """Simula múltiplos futuros possíveis.
        A predição para o PRÓXIMO step é a média do step 1 de todas as branches.
        A profundidade serve para avaliar qual velocidade é mais consistente."""
        branches = []
        predicoes_step1 = []
        for b in range(n_branches):
            pos = estado_atual
            vel = velocidade + random.gauss(0, 0.1)
            trajetoria = [pos]
            for step in range(profundidade):
                pos += vel + random.gauss(0, 0.05)
                trajetoria.append(round(pos, 4))
                self.t_explorados += 1
            predicoes_step1.append(trajetoria[1])  # Predição para step+1
            branches.append({
                "vel_usada": round(vel, 4),
                "pos_step1": round(trajetoria[1], 4),
                "pos_final": round(pos, 4),
            })

        # Predição para próximo step: média das predições step+1
        media_step1 = sum(predicoes_step1) / len(predicoes_step1)
        return {
            "n_branches": n_branches,
            "profundidade": profundidade,
            "timesteps_explorados": n_branches * profundidade,
            "predicao_media": round(media_step1, 4),
            "branches_resumo": branches,
        }


class DualClockSystem:
    """Sistema que roda os dois clocks sincronizados."""
    def __init__(self, n_branches: int = 5, profundidade: int = 20):
        self.c1 = ClockInercial()
        self.c2 = ClockProspectivo()
        self.n_branches = n_branches
        self.profundidade = profundidade
        self.log: List[Dict] = []

    def run(self, n_steps_reais: int = 50):
        for step in range(n_steps_reais):
            # Clock 2: explora futuros ANTES do dado real chegar
            exploracao = self.c2.explorar(
                self.c1.estado, self.c1.velocidade,
                self.n_branches, self.profundidade
            )

            # Clock 1: avança 1 step com dado real
            estado_real = self.c1.tick()

            # Calcular erro da predição
            erro = abs(exploracao["predicao_media"] - estado_real)

            self.log.append({
                "step": step,
                "real": round(estado_real, 4),
                "predicao": exploracao["predicao_media"],
                "erro": round(erro, 4),
                "timesteps_explorados_total": self.c2.t_explorados,
            })

        return self.log


class SingleClockSystem:
    """Baseline: apenas Clock 1, sem exploração prospectiva."""
    def __init__(self):
        self.c1 = ClockInercial()
        self.log = []
        self.predicao_simples = 0.0

    def run(self, n_steps: int = 50):
        for step in range(n_steps):
            # Predição ingênua: estado atual + velocidade
            self.predicao_simples = self.c1.estado + self.c1.velocidade
            estado_real = self.c1.tick()
            erro = abs(self.predicao_simples - estado_real)
            self.log.append({
                "step": step,
                "real": round(estado_real, 4),
                "predicao": round(self.predicao_simples, 4),
                "erro": round(erro, 4),
            })
        return self.log


def main():
    print("=" * 60)
    print("  LAB12 — DUAL CLOCK (TEORIA-F 12D)")
    print("=" * 60)

    res = {"meta": {"timestamp": datetime.now().isoformat(), "seed": SEED},
           "experimentos": {}}

    N_STEPS = 100

    # ── Dual Clock ────────────────────────────────────────
    print("\n  ▶ Dual Clock (5 branches × 20 profundidade)...")
    random.seed(SEED)
    dual = DualClockSystem(n_branches=5, profundidade=20)
    log_dual = dual.run(N_STEPS)
    erro_medio_dual = sum(l["erro"] for l in log_dual) / len(log_dual)
    t_explorados = log_dual[-1]["timesteps_explorados_total"]

    print(f"    Erro médio: {erro_medio_dual:.4f}")
    print(f"    Timesteps explorados: {t_explorados:,}")
    print(f"    Ratio: Clock2 explorou {t_explorados // N_STEPS}x mais steps que Clock1")

    # ── Single Clock ──────────────────────────────────────
    print("\n  ▶ Single Clock (baseline)...")
    random.seed(SEED)
    single = SingleClockSystem()
    log_single = single.run(N_STEPS)
    erro_medio_single = sum(l["erro"] for l in log_single) / len(log_single)

    print(f"    Erro médio: {erro_medio_single:.4f}")

    # ── Comparação ────────────────────────────────────────
    melhoria = (1 - erro_medio_dual / max(0.0001, erro_medio_single)) * 100
    print(f"\n  ─── Comparação ───")
    print(f"    Dual:   {erro_medio_dual:.4f}")
    print(f"    Single: {erro_medio_single:.4f}")
    print(f"    Melhoria: {melhoria:.1f}%")
    print(f"    ✅ Dual melhor que Single: {'SIM' if erro_medio_dual < erro_medio_single else 'NÃO'}")
    print(f"    ✅ Clock2 >100 steps à frente: {'SIM' if t_explorados / N_STEPS > 100 else 'NÃO'}")

    res["experimentos"] = {
        "dual_clock": {
            "erro_medio": round(erro_medio_dual, 6),
            "timesteps_explorados": t_explorados,
            "ratio_exploracao": t_explorados // N_STEPS,
            "n_steps_reais": N_STEPS,
        },
        "single_clock": {
            "erro_medio": round(erro_medio_single, 6),
        },
        "comparacao": {
            "melhoria_pct": round(melhoria, 1),
            "dual_melhor": erro_medio_dual < erro_medio_single,
            "criterio_100x": t_explorados / N_STEPS > 100,
        }
    }

    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    out = os.path.join(RESULTADOS_DIR, 'lab12_results.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Salvo em: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
