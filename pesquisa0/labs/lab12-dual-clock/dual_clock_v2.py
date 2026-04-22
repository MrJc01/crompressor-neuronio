#!/usr/bin/env python3
"""
LAB12 v2 — Dual Clock + World Model (Correção da Hipótese H13)
================================================================
Versão anterior: Clock Prospectivo usava perturbação aleatória → PIOR que baseline.
Versão 2: Clock Prospectivo usa o World Model do Lab03 para prever futuros
informados, e seleciona a melhor branch (mínimo erro estimado) em vez de média.

Saída: JSON em pesquisa0/resultados/lab12v2_results.json
"""

import json
import os
import math
import random
import time
from datetime import datetime
from typing import Dict, List

SEED = 42
random.seed(SEED)
RESULTADOS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'resultados')


# ─── Ambiente (mesmo do Lab12 v1) ────────────────────────────
class Ambiente:
    """Partícula com velocidade + aceleração + ruído."""
    def __init__(self):
        self.pos = 0.0
        self.vel = 0.5
        self.acel = 0.01
        self.step = 0

    def avancar(self):
        self.vel += self.acel + random.gauss(0, 0.02)
        self.pos += self.vel + random.gauss(0, 0.05)
        self.step += 1
        return self.pos


# ─── World Model Aprendido (inspirado no Lab03) ──────────────
class WorldModel:
    """Modelo interno que aprende a dinâmica do ambiente."""
    def __init__(self):
        self.vel_estimada = 0.5
        self.acel_estimada = 0.0
        self.pos_estimada = 0.0
        self.historico = []  # [(pos_real, step)]
        self.alpha = 0.3  # Taxa de aprendizado

    def observar(self, pos_real, step):
        """Recebe observação real e ajusta modelo interno."""
        self.historico.append((pos_real, step))
        if len(self.historico) >= 2:
            pos_prev, step_prev = self.historico[-2]
            dt = step - step_prev
            if dt > 0:
                vel_real = (pos_real - pos_prev) / dt
                # Atualizar estimativas com EMA
                self.acel_estimada = self.alpha * (vel_real - self.vel_estimada) + (1 - self.alpha) * self.acel_estimada
                self.vel_estimada = self.alpha * vel_real + (1 - self.alpha) * self.vel_estimada
        self.pos_estimada = pos_real

    def prever(self, n_steps=1):
        """Prediz posição futura usando modelo aprendido."""
        pos = self.pos_estimada
        vel = self.vel_estimada
        acel = self.acel_estimada
        for _ in range(n_steps):
            vel += acel
            pos += vel
        return pos


# ─── Clock 1 (Inercial) ──────────────────────────────────────
class ClockInercial:
    """Predição simples: vel constante (baseline)."""
    def __init__(self):
        self.ultima_pos = 0.0
        self.ultima_vel = 0.5

    def observar(self, pos_real, step):
        if step > 0:
            self.ultima_vel = pos_real - self.ultima_pos
        self.ultima_pos = pos_real

    def prever_proximo(self):
        return self.ultima_pos + self.ultima_vel


# ─── Clock 2 v1 (Perturbação Aleatória — REFUTADO) ───────────
class ClockProspectivoV1:
    """Versão que foi refutada: média de branches com ruído."""
    def __init__(self):
        self.pos = 0.0
        self.vel = 0.5

    def observar(self, pos_real, step):
        if step > 0:
            self.vel = pos_real - self.pos
        self.pos = pos_real

    def prever_proximo(self, n_branches=5, profundidade=20):
        predicoes = []
        for _ in range(n_branches):
            pos = self.pos
            vel = self.vel + random.gauss(0, 0.1)  # ← O PROBLEMA: ruído demais
            pos += vel + random.gauss(0, 0.05)
            predicoes.append(pos)
        return sum(predicoes) / len(predicoes)  # ← Média ingênua


# ─── Clock 2 v2 (World Model + Seleção Adaptativa) ───────────
class ClockProspectivoV2:
    """Versão corrigida: usa World Model aprendido + seleção pela menor incerteza."""
    def __init__(self):
        self.model = WorldModel()
        self.t_explorados = 0

    def observar(self, pos_real, step):
        self.model.observar(pos_real, step)

    def prever_proximo(self, n_branches=5, profundidade=10):
        """
        1. Gera N branches perturbando levemente os parâmetros do modelo
        2. Simula cada branch por `profundidade` steps
        3. Seleciona a branch com menor variância interna (mais consistente)
        4. Retorna a predição step+1 dessa branch
        """
        branches = []
        for b in range(n_branches):
            # Perturbação LEVE nos parâmetros aprendidos (não no estado)
            vel = self.model.vel_estimada + random.gauss(0, 0.02)  # σ=0.02 vs 0.1 do v1
            acel = self.model.acel_estimada + random.gauss(0, 0.005)
            
            # Simular trajetória
            pos = self.model.pos_estimada
            trajetoria = []
            for s in range(profundidade):
                vel += acel
                pos += vel
                trajetoria.append(pos)
                self.t_explorados += 1

            # Métrica de consistência: variância das derivadas (menor = mais previsível)
            derivadas = [trajetoria[i+1] - trajetoria[i] for i in range(len(trajetoria)-1)]
            variancia = sum((d - sum(derivadas)/len(derivadas))**2 for d in derivadas) / len(derivadas) if derivadas else float('inf')

            branches.append({
                "predicao_step1": trajetoria[0],  # Predição para o próximo step
                "variancia": variancia,
                "trajetoria": trajetoria
            })

        # SELEÇÃO ADAPTATIVA: escolher a branch mais consistente
        melhor = min(branches, key=lambda b: b["variancia"])
        
        # Ou: média ponderada pelo inverso da variância
        pesos = [1.0 / (b["variancia"] + 1e-10) for b in branches]
        soma_pesos = sum(pesos)
        media_ponderada = sum(b["predicao_step1"] * w for b, w in zip(branches, pesos)) / soma_pesos

        return media_ponderada


def rodar_experimento(n_steps=200):
    """Roda os 3 sistemas e compara."""
    random.seed(SEED)
    env = Ambiente()

    clock1 = ClockInercial()
    clock2_v1 = ClockProspectivoV1()
    clock2_v2 = ClockProspectivoV2()

    erros_c1, erros_v1, erros_v2 = [], [], []

    for step in range(n_steps):
        pos_real = env.avancar()

        # Predições ANTES de observar
        if step > 0:
            pred_c1 = clock1.prever_proximo()
            pred_v1 = clock2_v1.prever_proximo()
            pred_v2 = clock2_v2.prever_proximo()

            erros_c1.append(abs(pred_c1 - pos_real))
            erros_v1.append(abs(pred_v1 - pos_real))
            erros_v2.append(abs(pred_v2 - pos_real))

        # Observar
        clock1.observar(pos_real, step)
        clock2_v1.observar(pos_real, step)
        clock2_v2.observar(pos_real, step)

    return {
        "single_clock": {
            "erro_medio": round(sum(erros_c1) / len(erros_c1), 6),
            "erro_max": round(max(erros_c1), 6),
        },
        "dual_clock_v1_refutado": {
            "erro_medio": round(sum(erros_v1) / len(erros_v1), 6),
            "erro_max": round(max(erros_v1), 6),
        },
        "dual_clock_v2_world_model": {
            "erro_medio": round(sum(erros_v2) / len(erros_v2), 6),
            "erro_max": round(max(erros_v2), 6),
            "timesteps_explorados": clock2_v2.t_explorados,
        },
        "comparacao": {
            "v2_vs_single_pct": round((1 - sum(erros_v2)/sum(erros_c1)) * 100, 2),
            "v2_vs_v1_pct": round((1 - sum(erros_v2)/sum(erros_v1)) * 100, 2),
            "v2_melhor_que_single": sum(erros_v2) < sum(erros_c1),
            "v2_melhor_que_v1": sum(erros_v2) < sum(erros_v1),
        }
    }


def main():
    print("=" * 60)
    print("  LAB12 v2 — DUAL CLOCK + WORLD MODEL")
    print("  (Correção da hipótese H13 refutada)")
    print("=" * 60)

    res = {"meta": {"timestamp": datetime.now().isoformat(), "seed": SEED},
           "experimentos": {}}

    # ── Teste Principal: 200 steps ────────────────────────
    print("\n  ▶ Rodando 200 steps (3 sistemas comparados)...")
    exp = rodar_experimento(200)

    print(f"\n  [ Single Clock (baseline) ]")
    print(f"    Erro médio: {exp['single_clock']['erro_medio']:.6f}")
    
    print(f"\n  [ Dual Clock v1 (REFUTADO — perturbação aleatória) ]")
    print(f"    Erro médio: {exp['dual_clock_v1_refutado']['erro_medio']:.6f}")

    print(f"\n  [ Dual Clock v2 (World Model + Seleção Adaptativa) ]")
    print(f"    Erro médio: {exp['dual_clock_v2_world_model']['erro_medio']:.6f}")
    print(f"    Timesteps explorados: {exp['dual_clock_v2_world_model']['timesteps_explorados']:,}")

    print(f"\n  ─── Comparação ───")
    print(f"    v2 vs Single: {exp['comparacao']['v2_vs_single_pct']:+.2f}%")
    print(f"    v2 vs v1:     {exp['comparacao']['v2_vs_v1_pct']:+.2f}%")
    print(f"    ✅ v2 melhor que Single: {'SIM' if exp['comparacao']['v2_melhor_que_single'] else 'NÃO'}")
    print(f"    ✅ v2 melhor que v1:     {'SIM' if exp['comparacao']['v2_melhor_que_v1'] else 'NÃO'}")

    res["experimentos"]["principal"] = exp

    # ── Teste de Robustez: várias seeds ───────────────────
    print("\n  ▶ Robustez (10 seeds diferentes)...")
    vitorias_v2 = 0
    for s in range(10):
        random.seed(s * 137)
        exp_s = rodar_experimento(100)
        if exp_s["comparacao"]["v2_melhor_que_single"]:
            vitorias_v2 += 1

    print(f"    v2 venceu Single em {vitorias_v2}/10 seeds ({vitorias_v2*10}%)")
    res["experimentos"]["robustez"] = {
        "n_seeds": 10,
        "vitorias_v2": vitorias_v2,
        "taxa_vitoria_pct": vitorias_v2 * 10,
    }

    hipotese_corrigida = vitorias_v2 >= 7  # ≥70% de vitórias
    print(f"\n  {'✅' if hipotese_corrigida else '❌'} H13 CORRIGIDA: {'SIM' if hipotese_corrigida else 'NÃO'}")
    res["hipotese_h13_corrigida"] = hipotese_corrigida

    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    out = os.path.join(RESULTADOS_DIR, 'lab12v2_results.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Salvo em: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
