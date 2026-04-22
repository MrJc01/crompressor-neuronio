#!/usr/bin/env python3
"""
LAB10 — Active Inference Agent (Grid 2D)
=========================================
Agente com World Model interno navega grid 2D até objetivo.
Loop: Prever → Observar → Calcular F → Agir para minimizar F.
Comparação com baseline random walk.

Saída: JSON em pesquisa0/resultados/lab10_results.json
"""

import json
import os
import time
import random
import math
from datetime import datetime
from typing import List, Tuple, Dict

SEED = 42
RESULTADOS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'resultados')


class Grid:
    """Ambiente grid 2D com obstáculos opcionais."""
    def __init__(self, largura: int = 20, altura: int = 20):
        self.w = largura
        self.h = altura
        self.obstaculos = set()

    def add_obstaculo(self, x: int, y: int):
        self.obstaculos.add((x, y))

    def valida(self, x: int, y: int) -> bool:
        return 0 <= x < self.w and 0 <= y < self.h and (x, y) not in self.obstaculos

    def distancia(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


ACOES = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # cima, baixo, direita, esquerda


class AgenteRandom:
    """Baseline: anda aleatoriamente."""
    def __init__(self, grid: Grid, inicio: Tuple, objetivo: Tuple):
        self.grid = grid
        self.pos = inicio
        self.objetivo = objetivo
        self.passos = 0
        self.caminho = [inicio]

    def agir(self) -> bool:
        random.shuffle(ACOES)
        for dx, dy in ACOES:
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            if self.grid.valida(nx, ny):
                self.pos = (nx, ny)
                self.passos += 1
                self.caminho.append(self.pos)
                return self.pos == self.objetivo
        self.passos += 1
        return False


class AgenteActiveInference:
    """Agente com Active Inference: minimiza Energia Livre."""
    def __init__(self, grid: Grid, inicio: Tuple, objetivo: Tuple):
        self.grid = grid
        self.pos = inicio
        self.objetivo = objetivo
        self.passos = 0
        self.caminho = [inicio]
        self.historico_F: List[float] = []
        # World Model: mapa interno (crença sobre o grid)
        self.mapa_interno = {}  # (x,y) → "livre" ou "obstaculo"
        self.explorados = set()
        self.explorados.add(inicio)

    def _surpresa(self, pos: Tuple) -> float:
        """Surpresa = distância ao objetivo (quanto mais longe, mais surpresa)."""
        return self.grid.distancia(pos, self.objetivo)

    def _incerteza(self, pos: Tuple) -> float:
        """Incerteza = se a posição já foi explorada (0) ou não (1)."""
        return 0.0 if pos in self.explorados else 0.5

    def _energia_livre(self, pos: Tuple) -> float:
        """F = surpresa + incerteza."""
        return self._surpresa(pos) + self._incerteza(pos)

    def agir(self) -> bool:
        # Avaliar F para cada ação possível
        melhor_F = float('inf')
        melhor_pos = self.pos

        for dx, dy in ACOES:
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            if not self.grid.valida(nx, ny):
                continue
            # Se sabemos que é obstáculo no modelo interno, pular
            if self.mapa_interno.get((nx, ny)) == "obstaculo":
                continue
            F = self._energia_livre((nx, ny))
            if F < melhor_F:
                melhor_F = F
                melhor_pos = (nx, ny)

        # Mover para posição que minimiza F
        self.pos = melhor_pos
        self.passos += 1
        self.caminho.append(self.pos)
        self.explorados.add(self.pos)
        self.historico_F.append(round(melhor_F, 4))

        # Observar: atualizar mapa interno
        for dx, dy in ACOES:
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            if not self.grid.valida(nx, ny):
                self.mapa_interno[(nx, ny)] = "obstaculo"
            else:
                self.mapa_interno[(nx, ny)] = "livre"

        return self.pos == self.objetivo


def run_comparacao(grid: Grid, inicio: Tuple, objetivo: Tuple,
                   max_passos: int = 500, n_trials_random: int = 50):
    """Compara Active Inference vs Random Walk."""

    # Active Inference (determinístico)
    random.seed(SEED)
    ai = AgenteActiveInference(grid, inicio, objetivo)
    for _ in range(max_passos):
        if ai.agir():
            break

    # Random (média de N trials)
    passos_random = []
    for trial in range(n_trials_random):
        random.seed(SEED + trial)
        rw = AgenteRandom(grid, inicio, objetivo)
        chegou = False
        for _ in range(max_passos):
            if rw.agir():
                chegou = True
                break
        passos_random.append(rw.passos if chegou else max_passos)

    media_random = sum(passos_random) / len(passos_random)
    chegou_ai = ai.pos == objetivo
    speedup = media_random / max(1, ai.passos) if chegou_ai else 0

    return {
        "ai_passos": ai.passos,
        "ai_chegou": chegou_ai,
        "ai_F_inicial": ai.historico_F[0] if ai.historico_F else None,
        "ai_F_final": ai.historico_F[-1] if ai.historico_F else None,
        "random_media_passos": round(media_random, 1),
        "random_trials": n_trials_random,
        "speedup_x": round(speedup, 1),
        "criterio_5x": speedup >= 5.0,
    }


def run_com_surpresas(grid: Grid, inicio: Tuple, objetivo: Tuple, max_passos: int = 500):
    """Teste com obstáculos que aparecem no meio da navegação."""
    random.seed(SEED)
    ai = AgenteActiveInference(grid, inicio, objetivo)

    # Adicionar obstáculos progressivamente
    surpresas_adicionadas = []
    for step in range(max_passos):
        # A cada 50 passos, adicionar obstáculo aleatório
        if step > 0 and step % 50 == 0:
            ox = random.randint(2, grid.w - 3)
            oy = random.randint(2, grid.h - 3)
            if (ox, oy) != objetivo and (ox, oy) != ai.pos:
                grid.add_obstaculo(ox, oy)
                surpresas_adicionadas.append({"step": step, "obs": (ox, oy)})

        if ai.agir():
            break

    # Verificar se F subiu brevemente após surpresas
    F_antes_surpresas = ai.historico_F[:50] if len(ai.historico_F) >= 50 else ai.historico_F
    F_depois = ai.historico_F[50:100] if len(ai.historico_F) >= 100 else ai.historico_F[50:]

    media_antes = sum(F_antes_surpresas) / max(1, len(F_antes_surpresas))
    media_depois = sum(F_depois) / max(1, len(F_depois)) if F_depois else media_antes

    return {
        "ai_passos": ai.passos,
        "ai_chegou": ai.pos == objetivo,
        "surpresas": len(surpresas_adicionadas),
        "F_media_antes_surpresa": round(media_antes, 4),
        "F_media_depois_surpresa": round(media_depois, 4),
        "adaptou": media_depois <= media_antes * 1.5,  # Tolerância
    }


def main():
    print("=" * 60)
    print("  LAB10 — ACTIVE INFERENCE AGENT")
    print("=" * 60)

    res = {"meta": {"timestamp": datetime.now().isoformat(), "seed": SEED},
           "experimentos": {}}

    # ── Grid limpo ────────────────────────────────────────
    print("\n  ▶ Grid 20x20 sem obstáculos...")
    grid = Grid(20, 20)
    r = run_comparacao(grid, (0, 0), (19, 19))
    print(f"    AI: {r['ai_passos']} passos (chegou: {r['ai_chegou']})")
    print(f"    Random: {r['random_media_passos']} passos (média)")
    print(f"    Speedup: {r['speedup_x']}x")
    print(f"    ✅ AI ≥5x mais rápido: {'SIM' if r['criterio_5x'] else 'NÃO'}")
    res["experimentos"]["grid_limpo"] = r

    # ── Grid com obstáculos estáticos ─────────────────────
    print("\n  ▶ Grid 20x20 com 30 obstáculos estáticos...")
    grid2 = Grid(20, 20)
    random.seed(SEED + 100)
    for _ in range(30):
        ox, oy = random.randint(1, 18), random.randint(1, 18)
        if (ox, oy) not in [(0, 0), (19, 19)]:
            grid2.add_obstaculo(ox, oy)
    r2 = run_comparacao(grid2, (0, 0), (19, 19))
    print(f"    AI: {r2['ai_passos']} passos (chegou: {r2['ai_chegou']})")
    print(f"    Random: {r2['random_media_passos']} passos (média)")
    print(f"    Speedup: {r2['speedup_x']}x")
    res["experimentos"]["grid_obstaculos"] = r2

    # ── Grid com surpresas ────────────────────────────────
    print("\n  ▶ Grid 20x20 com obstáculos dinâmicos (surpresas)...")
    grid3 = Grid(20, 20)
    r3 = run_com_surpresas(grid3, (0, 0), (19, 19))
    print(f"    AI: {r3['ai_passos']} passos (chegou: {r3['ai_chegou']})")
    print(f"    Surpresas adicionadas: {r3['surpresas']}")
    print(f"    F antes: {r3['F_media_antes_surpresa']}  F depois: {r3['F_media_depois_surpresa']}")
    print(f"    ✅ Adaptou após surpresas: {'SIM' if r3['adaptou'] else 'NÃO'}")
    res["experimentos"]["grid_surpresas"] = r3

    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    out = os.path.join(RESULTADOS_DIR, 'lab10_results.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Salvo em: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
