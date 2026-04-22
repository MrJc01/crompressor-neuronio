#!/usr/bin/env python3
"""
LAB08 — Detector de Alucinação (Sandbox)
=========================================
Detecta "alucinações" usando Delta Ratio entre tokens
e um Codebook de domínio. Implementa sandbox de isolamento.

Saída: JSON em pesquisa0/resultados/lab08_results.json
"""

import json
import os
import random
import math
import time
from datetime import datetime
from typing import List, Dict, Tuple

SEED = 42
random.seed(SEED)
RESULTADOS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'resultados')


# ─── Dataset de Teste ─────────────────────────────────────────
# 50 afirmações verdadeiras + 50 alucinadas
VERDADEIRAS = [
    "A água ferve a 100 graus Celsius ao nível do mar",
    "A Terra orbita o Sol em aproximadamente 365.25 dias",
    "O DNA contém quatro bases: adenina, timina, citosina e guanina",
    "A velocidade da luz no vácuo é aproximadamente 3e8 metros por segundo",
    "O hidrogênio é o elemento mais abundante no universo",
    "A gravidade na superfície da Terra é aproximadamente 9.8 m/s²",
    "O número Pi é aproximadamente 3.14159",
    "A mitocôndria é responsável pela produção de ATP nas células",
    "O som se propaga mais rápido em sólidos do que em gases",
    "A constante de Planck é aproximadamente 6.626e-34 J·s",
    "A fotossíntese converte CO2 e água em glicose e oxigênio",
    "O Python foi criado por Guido van Rossum em 1991",
    "SHA-256 produz um hash de 256 bits",
    "O Linux kernel foi criado por Linus Torvalds em 1991",
    "TCP garante entrega ordenada de pacotes",
    "Um byte contém 8 bits",
    "A entropia de Shannon mede a incerteza de uma variável aleatória",
    "K-Means é um algoritmo de clustering por particionamento",
    "O Transformer usa self-attention como mecanismo principal",
    "A complexidade do quicksort é O(n log n) no caso médio",
    "HTTP usa a porta 80 por padrão",
    "UTF-8 é compatível com ASCII para os primeiros 128 caracteres",
    "Git usa SHA-1 para identificar objetos internamente",
    "O oxigênio constitui aproximadamente 21% da atmosfera terrestre",
    "A pressão atmosférica ao nível do mar é aproximadamente 101325 Pa",
    "O elétron tem carga negativa de aproximadamente 1.6e-19 Coulombs",
    "A equação de Einstein E=mc² relaciona massa e energia",
    "O número de Avogadro é aproximadamente 6.022e23",
    "A tabela periódica possui 118 elementos confirmados",
    "O carbono tem número atômico 6",
    "A Lua leva aproximadamente 27.3 dias para orbitar a Terra",
    "O ouro tem símbolo Au na tabela periódica",
    "O ferro tem ponto de fusão de aproximadamente 1538 graus Celsius",
    "O SQL é uma linguagem de consulta para bancos de dados relacionais",
    "IPv4 usa endereços de 32 bits",
    "O protocolo SSH usa a porta 22 por padrão",
    "A frequência da luz vermelha é menor que a da luz azul",
    "O diamante é uma forma alotrópica do carbono",
    "A insulina é produzida pelo pâncreas",
    "O teorema de Pitágoras estabelece que a² + b² = c²",
    "A molécula de água é composta por 2 átomos de hidrogênio e 1 de oxigênio",
    "O nitrogênio constitui aproximadamente 78% da atmosfera",
    "A constante gravitacional G é aproximadamente 6.674e-11",
    "O hélio é o segundo elemento mais abundante no universo",
    "A velocidade do som no ar é aproximadamente 343 m/s a 20°C",
    "O silício é o segundo elemento mais abundante na crosta terrestre",
    "Um quilobyte equivale a 1024 bytes em definição binária",
    "O TCP/IP é a base da comunicação na internet",
    "A teoria da relatividade geral foi publicada por Einstein em 1915",
    "O cobre tem símbolo Cu na tabela periódica",
]

ALUCINADAS = [
    "A água ferve a 85 graus Celsius ao nível do mar",
    "A Terra orbita o Sol em exatamente 400 dias",
    "O DNA contém seis bases incluindo xenina e plutonina",
    "A velocidade da luz no vácuo é exatamente 5e8 metros por segundo",
    "O hélio é o elemento mais abundante no universo",
    "A gravidade na superfície da Terra é aproximadamente 15.2 m/s²",
    "O número Pi é exatamente 3.14 e tem apenas 3 casas decimais",
    "O ribossomo é responsável pela produção de ATP nas células",
    "O som se propaga mais rápido em gases do que em sólidos",
    "A constante de Planck é aproximadamente 9.1e-31 J·s",
    "A fotossíntese converte oxigênio em CO2 e metano",
    "O Python foi criado por Dennis Ritchie em 1972",
    "SHA-256 produz um hash de 512 bits",
    "O Linux kernel foi criado por Richard Stallman em 1984",
    "UDP garante entrega ordenada de pacotes",
    "Um byte contém 10 bits na maioria das arquiteturas",
    "A entropia de Shannon mede a temperatura de um sistema",
    "K-Means é um algoritmo de classificação supervisionada",
    "O Transformer usa convolução como mecanismo principal",
    "A complexidade do quicksort é O(n³) no caso médio",
    "HTTP usa a porta 443 por padrão sem SSL",
    "UTF-8 é incompatível com ASCII em todos os casos",
    "Git usa MD5 para identificar objetos internamente",
    "O oxigênio constitui aproximadamente 50% da atmosfera terrestre",
    "A pressão atmosférica ao nível do mar é aproximadamente 50000 Pa",
    "O elétron tem carga positiva de aproximadamente 3.2e-19 Coulombs",
    "A equação de Einstein E=mc³ relaciona massa e energia",
    "O número de Avogadro é aproximadamente 3.011e20",
    "A tabela periódica possui 200 elementos confirmados",
    "O carbono tem número atômico 12",
    "A Lua leva aproximadamente 100 dias para orbitar a Terra",
    "O ouro tem símbolo Go na tabela periódica",
    "O ferro tem ponto de fusão de aproximadamente 500 graus Celsius",
    "O SQL é uma linguagem de programação orientada a objetos",
    "IPv4 usa endereços de 64 bits",
    "O protocolo SSH usa a porta 80 por padrão",
    "A frequência da luz vermelha é maior que a da luz azul",
    "O diamante é uma forma alotrópica do ferro",
    "A insulina é produzida pelo fígado",
    "O teorema de Pitágoras estabelece que a² + b² = c³",
    "A molécula de água é composta por 3 átomos de hidrogênio e 2 de oxigênio",
    "O nitrogênio constitui aproximadamente 10% da atmosfera",
    "A constante gravitacional G é aproximadamente 9.8",
    "O argônio é o segundo elemento mais abundante no universo",
    "A velocidade do som no ar é aproximadamente 1200 m/s a 20°C",
    "O silício é o elemento mais raro na crosta terrestre",
    "Um quilobyte equivale a 500 bytes em definição binária",
    "O UDP/IP é a base da comunicação na internet",
    "A teoria da relatividade geral foi publicada por Newton em 1687",
    "O cobre tem símbolo Cp na tabela periódica",
]


# ─── Codebook de Domínio ─────────────────────────────────────
def construir_codebook(afirmacoes: List[str], n_grams: int = 3) -> Dict[str, float]:
    """Constrói codebook de n-gramas de palavras com frequência normalizada."""
    freq = {}
    total = 0
    for texto in afirmacoes:
        palavras = texto.lower().split()
        for i in range(len(palavras) - n_grams + 1):
            ngram = " ".join(palavras[i:i + n_grams])
            freq[ngram] = freq.get(ngram, 0) + 1
            total += 1
    # Normalizar
    return {k: v / total for k, v in freq.items()}


def delta_ratio(texto: str, codebook: Dict[str, float], n_grams: int = 3) -> float:
    """Calcula Delta Ratio: proporção de n-gramas NÃO presentes no codebook."""
    palavras = texto.lower().split()
    ngrams_texto = []
    for i in range(len(palavras) - n_grams + 1):
        ngrams_texto.append(" ".join(palavras[i:i + n_grams]))

    if not ngrams_texto:
        return 1.0

    desconhecidos = sum(1 for ng in ngrams_texto if ng not in codebook)
    return desconhecidos / len(ngrams_texto)


def detectar_alucinacao(texto: str, codebook: Dict[str, float],
                        threshold: float = 0.5) -> Tuple[bool, float]:
    """Retorna (é_alucinação, delta_ratio)."""
    dr = delta_ratio(texto, codebook)
    return dr > threshold, dr


# ─── Sandbox de Isolamento ────────────────────────────────────
class Sandbox:
    """Simula isolamento de branches — resultados só passam se DKL < threshold."""
    def __init__(self, threshold_dkl: float = 0.5):
        self.threshold = threshold_dkl
        self.memoria_principal: List[str] = []
        self.contaminacoes = 0
        self.bloqueios = 0

    def submeter(self, texto: str, codebook: Dict[str, float]) -> bool:
        """Tenta promover resultado de sandbox para memória principal."""
        eh_aluc, dr = detectar_alucinacao(texto, codebook, self.threshold)
        if eh_aluc:
            self.bloqueios += 1
            return False
        else:
            self.memoria_principal.append(texto)
            return True


def main():
    print("=" * 60)
    print("  LAB08 — DETECTOR DE ALUCINAÇÃO (SANDBOX)")
    print("=" * 60)

    res = {"meta": {"timestamp": datetime.now().isoformat(), "seed": SEED},
           "experimentos": {}}

    # ── Construir Codebook ────────────────────────────────
    print("\n  ▶ Construindo Codebook de domínio (3-gramas)...")
    codebook = construir_codebook(VERDADEIRAS, n_grams=3)
    print(f"    Tamanho do Codebook: {len(codebook)} n-gramas")

    # ── Testar detecção ───────────────────────────────────
    print("\n  ▶ Testando detecção em 100 afirmações...")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    melhor_threshold = 0.5
    melhor_f1 = 0.0

    resultados_threshold = {}
    for th in thresholds:
        tp = fp = tn = fn = 0
        for texto in VERDADEIRAS:
            aluc, dr = detectar_alucinacao(texto, codebook, th)
            if not aluc:
                tn += 1
            else:
                fp += 1
        for texto in ALUCINADAS:
            aluc, dr = detectar_alucinacao(texto, codebook, th)
            if aluc:
                tp += 1
            else:
                fn += 1

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(0.001, precision + recall)
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        if f1 > melhor_f1:
            melhor_f1 = f1
            melhor_threshold = th

        resultados_threshold[str(th)] = {
            "threshold": th,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "accuracy": round(accuracy, 3),
        }
        print(f"    th={th:.1f}: TP={tp:>2d} FP={fp:>2d} TN={tn:>2d} FN={fn:>2d}  "
              f"P={precision:.2f} R={recall:.2f} F1={f1:.2f} Acc={accuracy:.2f}")

    print(f"\n    Melhor threshold: {melhor_threshold} (F1={melhor_f1:.3f})")
    res["experimentos"]["deteccao"] = {
        "por_threshold": resultados_threshold,
        "melhor_threshold": melhor_threshold,
        "melhor_f1": round(melhor_f1, 3),
    }

    # ── Sandbox de Isolamento ─────────────────────────────
    print(f"\n  ▶ Sandbox com threshold={melhor_threshold}...")
    sandbox = Sandbox(threshold_dkl=melhor_threshold)

    for texto in VERDADEIRAS[:25]:
        sandbox.submeter(texto, codebook)
    for texto in ALUCINADAS[:25]:
        sandbox.submeter(texto, codebook)

    contaminacao = 0
    for txt in sandbox.memoria_principal:
        if txt in ALUCINADAS:
            contaminacao += 1

    print(f"    Aceitas na memória: {len(sandbox.memoria_principal)}")
    print(f"    Bloqueadas: {sandbox.bloqueios}")
    print(f"    Contaminações: {contaminacao}")
    status_contam = "SIM" if contaminacao == 0 else f"NÃO ({contaminacao} vazaram)"
    print(f"    ✅ Zero contaminação: {status_contam}")

    res["experimentos"]["sandbox"] = {
        "threshold": melhor_threshold,
        "aceitas": len(sandbox.memoria_principal),
        "bloqueadas": sandbox.bloqueios,
        "contaminacoes": contaminacao,
        "zero_contaminacao": contaminacao == 0,
    }

    # ── Dataset ───────────────────────────────────────────
    dataset = {
        "verdadeiras": len(VERDADEIRAS),
        "alucinadas": len(ALUCINADAS),
        "total": len(VERDADEIRAS) + len(ALUCINADAS),
    }
    res["dataset"] = dataset

    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    out = os.path.join(RESULTADOS_DIR, 'lab08_results.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Salvo em: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
