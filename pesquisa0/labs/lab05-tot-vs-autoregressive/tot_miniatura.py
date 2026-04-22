#!/usr/bin/env python3
"""
LAB05 — Tree of Thoughts vs Geração Autoregressiva (Miniatura)
=============================================================
Simula a arquitetura "Tree of Thoughts" (ToT) vs Geração Linear,
usando Delta Storage para branches e Pruning baseado em avaliação.

O "problema" é o Jogo do 24: chegar a 24 usando 4 números e +, -, *, /.
Como não temos um LLM garantido aqui, usamos um gerador sintético
que simula as tentativas de um LLM com alguma probabilidade de acerto.

Saída: JSON em pesquisa0/resultados/lab05_results.json
"""

import json
import os
import time
import random
from datetime import datetime
from typing import List, Dict, Tuple, Set

SEED = 42
random.seed(SEED)
RESULTADOS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'resultados')


class JogoDo24Simulador:
    """
    Simula um LLM tentando jogar o Jogo do 24.
    Estado: lista de números restantes.
    Ação: escolher dois números e uma operação, substituí-los pelo resultado.
    """
    
    OPERATIONS = ['+', '-', '*', '/']
    
    @staticmethod
    def gerar_pensamentos(estado: List[float], n_pensamentos: int = 3) -> List[Tuple[List[float], str]]:
        """Dado um estado, gera N próximos estados possíveis (pensamentos)."""
        pensamentos = []
        if len(estado) <= 1:
            return pensamentos
            
        for _ in range(n_pensamentos):
            # Escolher 2 números aleatórios
            idx1, idx2 = random.sample(range(len(estado)), 2)
            n1, n2 = estado[idx1], estado[idx2]
            op = random.choice(JogoDo24Simulador.OPERATIONS)
            
            res = None
            if op == '+': res = n1 + n2
            elif op == '-': res = n1 - n2
            elif op == '*': res = n1 * n2
            elif op == '/': res = n1 / n2 if n2 != 0 else None
            
            if res is not None:
                novo_estado = estado.copy()
                # Remover do maior pro menor pra não dar erro de índice
                for i in sorted([idx1, idx2], reverse=True):
                    novo_estado.pop(i)
                novo_estado.append(res)
                
                # Adicionamos "ruído" simulando alucinação/erro do LLM (10% chance)
                if random.random() < 0.10:
                    novo_estado[-1] += random.choice([-1, 1])
                    
                pensamentos.append((novo_estado, f"{n1} {op} {n2} = {novo_estado[-1]}"))
                
        return pensamentos

    @staticmethod
    def avaliar_estado(estado: List[float]) -> float:
        """
        Sistema 2: Avalia a promessa de um estado (0.0 a 1.0).
        Retorna 1.0 se chegou em 24. Retorna 0.0 se impossível (len=1 e != 24).
        """
        if len(estado) == 1:
            return 1.0 if abs(estado[0] - 24.0) < 1e-5 else 0.0
            
        # Heurística simples: se a soma de todos os números está muito distante de 24, score menor.
        # Não é perfeito, mas simula a avaliação de "certeza" de um LLM.
        dist = abs(sum(estado) - 24.0)
        score = max(0.1, 1.0 - (dist / 100.0))
        return score


class AgenteAutoregressivo:
    """Gera o resultado linearmente (1 pensamento por vez, sem backtracking)."""
    def resolver(self, estado_inicial: List[float], max_steps: int = 3) -> Dict:
        estado = estado_inicial.copy()
        passos = []
        
        t0 = time.perf_counter()
        for _ in range(max_steps):
            if len(estado) <= 1:
                break
                
            # Na geração autoregressiva (greedy), ele escolhe a primeira ideia
            pensamentos = JogoDo24Simulador.gerar_pensamentos(estado, n_pensamentos=1)
            if not pensamentos:
                break
                
            estado, passo_str = pensamentos[0]
            passos.append(passo_str)
            
        t_total = time.perf_counter() - t0
        
        sucesso = (len(estado) == 1 and abs(estado[0] - 24.0) < 1e-5)
        return {
            "sucesso": sucesso,
            "passos": passos,
            "estado_final": estado,
            "tempo_ms": t_total * 1000,
            "nos_explorados": len(passos)
        }


class AgenteTreeOfThoughts:
    """Gera múltiplos pensamentos e busca o melhor caminho (BFS com poda)."""
    def resolver(self, estado_inicial: List[float], max_steps: int = 3,
                 b_pensamentos: int = 3, threshold_poda: float = 0.3) -> Dict:
        # Estado: (lista_numeros, lista_passos)
        fronteira = [(estado_inicial, [])]
        nos_explorados = 0
        
        t0 = time.perf_counter()
        
        for step in range(max_steps):
            nova_fronteira = []
            
            for estado, caminho in fronteira:
                # Se já é final, não gera mais
                if len(estado) <= 1:
                    nova_fronteira.append((estado, caminho, JogoDo24Simulador.avaliar_estado(estado)))
                    continue
                    
                # Gerar B branches
                pensamentos = JogoDo24Simulador.gerar_pensamentos(estado, n_pensamentos=b_pensamentos)
                for novo_est, passo_str in pensamentos:
                    nos_explorados += 1
                    score = JogoDo24Simulador.avaliar_estado(novo_est)
                    # Poda (pruning): só mantém se score >= threshold
                    if score >= threshold_poda:
                        nova_fronteira.append((novo_est, caminho + [passo_str], score))
                        
            # Ordenar por score e manter apenas as top-B (beam search) para não explodir a memória
            nova_fronteira.sort(key=lambda x: x[2], reverse=True)
            fronteira = [(e, c) for e, c, s in nova_fronteira[:b_pensamentos]]
            
            if not fronteira:
                break
                
        t_total = time.perf_counter() - t0
        
        # Verificar o melhor resultado final
        melhor_caminho = None
        sucesso = False
        estado_final = None
        
        for estado, caminho in fronteira:
            if len(estado) == 1 and abs(estado[0] - 24.0) < 1e-5:
                sucesso = True
                melhor_caminho = caminho
                estado_final = estado
                break
                
        if not sucesso and fronteira:
            estado_final = fronteira[0][0]
            melhor_caminho = fronteira[0][1]
            
        return {
            "sucesso": sucesso,
            "passos": melhor_caminho if melhor_caminho else [],
            "estado_final": estado_final if estado_final else [],
            "tempo_ms": t_total * 1000,
            "nos_explorados": nos_explorados
        }


def run_benchmark(n_jogos: int = 100):
    """Compara Autoregressivo vs ToT em N jogos do 24 aleatórios."""
    jogos = []
    for _ in range(n_jogos):
        jogos.append([random.randint(1, 13) for _ in range(4)])
        
    ar_sucessos = 0
    tot_sucessos = 0
    ar_tempo_total = 0
    tot_tempo_total = 0
    ar_nos_total = 0
    tot_nos_total = 0
    
    agente_ar = AgenteAutoregressivo()
    agente_tot = AgenteTreeOfThoughts()
    
    for jogo in jogos:
        random.seed(sum(jogo) + SEED) # Mesma seed para os dois no mesmo jogo
        
        res_ar = agente_ar.resolver(jogo)
        if res_ar["sucesso"]: ar_sucessos += 1
        ar_tempo_total += res_ar["tempo_ms"]
        ar_nos_total += res_ar["nos_explorados"]
        
        random.seed(sum(jogo) + SEED)
        res_tot = agente_tot.resolver(jogo, b_pensamentos=5)
        if res_tot["sucesso"]: tot_sucessos += 1
        tot_tempo_total += res_tot["tempo_ms"]
        tot_nos_total += res_tot["nos_explorados"]
        
    return {
        "n_jogos": n_jogos,
        "autoregressivo": {
            "taxa_sucesso": ar_sucessos / n_jogos,
            "tempo_medio_ms": round(ar_tempo_total / n_jogos, 2),
            "nos_explorados_media": round(ar_nos_total / n_jogos, 2)
        },
        "tree_of_thoughts": {
            "taxa_sucesso": tot_sucessos / n_jogos,
            "tempo_medio_ms": round(tot_tempo_total / n_jogos, 2),
            "nos_explorados_media": round(tot_nos_total / n_jogos, 2)
        },
        "comparacao": {
            "ganho_accuracy_pct": round((tot_sucessos - ar_sucessos) / max(1, ar_sucessos) * 100, 2),
            "overhead_computacional_x": round(max(1, tot_nos_total) / max(1, ar_nos_total), 2)
        }
    }


def main():
    print("=" * 60)
    print("  LAB05 — TREE OF THOUGHTS VS AUTOREGRESSIVO")
    print("=" * 60)

    res = {
        "meta": {"timestamp": datetime.now().isoformat(), "seed": SEED},
        "experimentos": {}
    }

    print("\n  ▶ Rodando benchmark no Jogo do 24 (1000 trials)...")
    benchmark = run_benchmark(n_jogos=1000)
    
    print(f"\n  [ Autoregressivo (Linear) ]")
    print(f"    Taxa de Sucesso: {benchmark['autoregressivo']['taxa_sucesso']*100:.1f}%")
    print(f"    Nós Explorados:  {benchmark['autoregressivo']['nos_explorados_media']:.1f} por jogo")
    
    print(f"\n  [ Tree of Thoughts (5 Branches + Pruning) ]")
    print(f"    Taxa de Sucesso: {benchmark['tree_of_thoughts']['taxa_sucesso']*100:.1f}%")
    print(f"    Nós Explorados:  {benchmark['tree_of_thoughts']['nos_explorados_media']:.1f} por jogo")
    
    print(f"\n  [ Comparação ]")
    ganho_acc = benchmark['comparacao']['ganho_accuracy_pct']
    overhead = benchmark['comparacao']['overhead_computacional_x']
    
    print(f"    Ganho de Accuracy: +{ganho_acc}%")
    print(f"    Overhead: {overhead}x mais computação")
    
    print(f"    ✅ ToT melhor que AR: {'SIM' if benchmark['tree_of_thoughts']['taxa_sucesso'] > benchmark['autoregressivo']['taxa_sucesso'] else 'NÃO'}")

    res["experimentos"]["benchmark_24_game"] = benchmark

    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    out = os.path.join(RESULTADOS_DIR, 'lab05_results.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Salvo em: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
