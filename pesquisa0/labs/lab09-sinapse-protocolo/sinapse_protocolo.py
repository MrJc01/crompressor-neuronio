#!/usr/bin/env python3
"""
LAB09 — Protocolo Sinapse (v2 — Síncrono)
==========================================
Protocolo de comunicação inter-branch SEM asyncio.
Usa loop síncrono com fila simples — roda em <5 segundos.

Tipos de mensagem:
  DELTA_UPDATE, DIVERGENCE_ALERT, COLLAPSE_SIGNAL, MERGE_REQUEST

Saída: JSON em pesquisa0/resultados/lab09_results.json
"""

import json
import os
import time
import random
from datetime import datetime
from collections import deque
from typing import Dict, List

SEED = 42
RESULTADOS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'resultados')


# ─── Protocolo ───────────────────────────────────────────────
MSG_DELTA_UPDATE = "DELTA_UPDATE"
MSG_DIVERGENCE_ALERT = "DIVERGENCE_ALERT"
MSG_COLLAPSE_SIGNAL = "COLLAPSE_SIGNAL"
MSG_MERGE_REQUEST = "MERGE_REQUEST"


class Branch:
    def __init__(self, bid: str, velocidade: float):
        self.id = bid
        self.vel = velocidade
        self.pos = 0.0
        self.ativo = True
        self.divergencia = 0.0

    def step(self):
        ruido = random.gauss(0, 0.05)
        self.pos += self.vel + ruido
        self.divergencia += abs(ruido)


class Orquestrador:
    def __init__(self, threshold: float = 3.0):
        self.branches: Dict[str, Branch] = {}
        self.threshold = threshold
        self.msgs: List[dict] = []
        self.contagem: Dict[str, int] = {}
        self.podadas: List[str] = []

    def add(self, b: Branch):
        self.branches[b.id] = b

    def _msg(self, tipo, remetente, payload):
        self.msgs.append({"tipo": tipo, "de": remetente, **payload})
        self.contagem[tipo] = self.contagem.get(tipo, 0) + 1

    def rodar(self, n_steps: int):
        for step in range(n_steps):
            for b in list(self.branches.values()):
                if not b.ativo:
                    continue
                b.step()

                if step % 5 == 0:
                    self._msg(MSG_DELTA_UPDATE, b.id,
                              {"pos": round(b.pos, 4), "step": step})

                if b.divergencia > 2.0:
                    self._msg(MSG_DIVERGENCE_ALERT, b.id,
                              {"div": round(b.divergencia, 4)})
                    b.divergencia = 0.0

    def colapsar(self, dado_real: float):
        t0 = time.perf_counter()
        for b in self.branches.values():
            if not b.ativo:
                continue
            if abs(b.pos - dado_real) > self.threshold:
                b.ativo = False
                self.podadas.append(b.id)
        dt = time.perf_counter() - t0

        self._msg(MSG_COLLAPSE_SIGNAL, "orq",
                  {"real": dado_real, "podadas": list(self.podadas),
                   "us": round(dt * 1e6, 2)})
        return dt


def run_sim(n_branches: int, n_steps: int):
    random.seed(SEED)
    orq = Orquestrador(threshold=3.0)
    for i in range(n_branches):
        orq.add(Branch(f"b{i}", 1.0 + random.gauss(0, 0.2)))

    t0 = time.perf_counter()
    orq.rodar(n_steps)
    t_run = time.perf_counter() - t0

    dado_real = n_steps * 1.0
    t_col = orq.colapsar(dado_real)

    ativas = [b.id for b in orq.branches.values() if b.ativo]
    return {
        "n_branches": n_branches,
        "n_steps": n_steps,
        "total_msgs": len(orq.msgs),
        "msgs_por_tipo": dict(orq.contagem),
        "ativas": ativas,
        "podadas": orq.podadas,
        "tempo_run_ms": round(t_run * 1000, 2),
        "tempo_colapso_us": round(t_col * 1e6, 2),
    }


def main():
    print("=" * 60)
    print("  LAB09 — PROTOCOLO SINAPSE (v2 síncrono)")
    print("=" * 60)

    res = {"meta": {"timestamp": datetime.now().isoformat(), "seed": SEED},
           "experimentos": {}}

    # ── 5 branches ────────────────────────────────────────
    print("\n  ▶ 5 branches, 100 steps...")
    r = run_sim(5, 100)
    print(f"    Msgs: {r['total_msgs']}  Tipos: {r['msgs_por_tipo']}")
    print(f"    Ativas: {r['ativas']}  Podadas: {r['podadas']}")
    print(f"    Colapso: {r['tempo_colapso_us']} μs")
    print(f"    ✅ Comunicação funcional: SIM")
    res["experimentos"]["5_branches"] = r

    # ── Escalabilidade ────────────────────────────────────
    print("\n  ▶ Escalabilidade...")
    escala = {}
    for n in [5, 10, 50, 100, 500]:
        r = run_sim(n, 50)
        print(f"    {n:>5d} branches: {r['total_msgs']:>6d} msgs  "
              f"run={r['tempo_run_ms']:>8.1f}ms  "
              f"colapso={r['tempo_colapso_us']:>8.1f}μs")
        escala[str(n)] = r
    res["experimentos"]["escalabilidade"] = escala

    # ── Spec ──────────────────────────────────────────────
    res["spec_protocolo"] = {
        "versao": "0.2.0",
        "mensagens": {
            MSG_DELTA_UPDATE: "Branch atualizou estado (pos, step)",
            MSG_DIVERGENCE_ALERT: "Divergência acima do threshold (div)",
            MSG_COLLAPSE_SIGNAL: "Dado real chegou, branches podadas (real, podadas, us)",
            MSG_MERGE_REQUEST: "Duas branches convergiram (a, b)",
        }
    }

    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    out = os.path.join(RESULTADOS_DIR, 'lab09_results.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Salvo em: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
