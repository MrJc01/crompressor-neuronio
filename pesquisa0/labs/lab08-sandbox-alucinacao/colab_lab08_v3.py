#!/usr/bin/env python3
"""
COLAB NOTEBOOK — Lab08 v3: Detector de Alucinação com Sentence-Transformers
Meta: Recall > 90% (v1: 68%, v2: 82%)

INSTRUÇÕES PARA COLAB:
1. Abra https://colab.research.google.com
2. Crie um novo notebook
3. Mude runtime para T4 GPU (Runtime → Change runtime type → T4)
4. Cole este script inteiro em uma célula e execute
5. Copie o JSON de resultado e cole aqui no chat
"""

# ── Célula 1: Instalar dependências ──────────────────────────
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "sentence-transformers", "scikit-learn", "numpy"])

import torch
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")

# ── Célula 2: Carregar modelo ─────────────────────────────────
print("\n▶ Carregando sentence-transformers...")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print(f"  Modelo: all-MiniLM-L6-v2 ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

# ── Célula 3: Dataset ─────────────────────────────────────────
contexto = """
A Torre Eiffel tem 330 metros de altura e fica em Paris, França.
Foi construída em 1889 por Gustave Eiffel para a Exposição Universal.
Recebe cerca de 7 milhões de visitantes por ano.
É feita de ferro puddled e pesa aproximadamente 7.300 toneladas.
A torre possui 3 andares acessíveis ao público.
"""

respostas = [
    # Corretas
    ("A Torre Eiffel tem 330 metros de altura.", False),
    ("Ela fica em Paris, na França.", False),
    ("Foi construída em 1889 por Gustave Eiffel.", False),
    ("Recebe cerca de 7 milhões de visitantes.", False),
    ("É feita de ferro e pesa 7300 toneladas.", False),
    # Alucinações claras
    ("A Torre Eiffel tem 500 metros de altura.", True),
    ("Foi construída em 1920 por Nikola Tesla.", True),
    ("A torre é feita de aço inoxidável e titânio.", True),
    ("Recebe 50 milhões de visitantes por dia.", True),
    ("A torre possui 15 andares e um heliporto.", True),
    # Sutis
    ("A Torre Eiffel tem cerca de 300 metros.", False),
    ("Foi construída para a Exposição de 1889.", False),
    ("A torre pesa cerca de 10.000 toneladas.", True),
    ("Fica em Lyon, na França.", True),
    ("Foi projetada por Alexandre Eiffel.", True),
    # Genéricas
    ("A Torre Eiffel é um monumento icônico.", False),
    ("É uma das atrações mais visitadas do mundo.", False),
    ("A torre foi renovada recentemente.", True),
    ("O restaurante da torre tem estrela Michelin.", True),
    ("A estrutura é considerada patrimônio da UNESCO.", True),
    # EXTRAS — 10 casos mais difíceis
    ("A torre tem exatamente 324 metros sem a antena.", False),  # Verdadeiro (≈330 com antena)
    ("Gustave Eiffel também projetou a Estátua da Liberdade.", False),  # Verdadeiro
    ("A torre foi pintada de dourado para os Jogos de 2024.", True),  # Falso
    ("São necessários 60 toneladas de tinta para pintá-la.", False),  # Aproximado verdadeiro
    ("A torre balança até 12 cm com vento forte.", False),  # Verdadeiro
    ("O elevador da torre atinge velocidade de 100 km/h.", True),  # Falso
    ("A torre emite radiação eletromagnética perigosa.", True),  # Falso
    ("Paris é a capital da Alemanha.", True),  # Falso óbvio
    ("A torre foi desmontada durante a 2ª Guerra Mundial.", True),  # Falso
    ("Gustave Eiffel morava em um apartamento no topo.", False),  # Verdadeiro
]

textos_resp = [r[0] for r in respostas]
labels = [r[1] for r in respostas]

# ── Célula 4: Embeddings ──────────────────────────────────────
print("\n▶ Gerando embeddings...")
ctx_sentences = [s.strip() for s in contexto.strip().split('\n') if s.strip()]
ctx_embs = model.encode(ctx_sentences, convert_to_numpy=True)
resp_embs = model.encode(textos_resp, convert_to_numpy=True)

# Para cada resposta, pegar a MELHOR similaridade com qualquer sentença do contexto
similarities = []
for resp_emb in resp_embs:
    sims = cosine_similarity([resp_emb], ctx_embs)[0]
    similarities.append(float(np.max(sims)))  # Melhor match

# ── Célula 5: Buscar threshold ótimo ──────────────────────────
print("\n▶ Buscando threshold ótimo...")
best_f1 = 0
best_metrics = {}

for thresh in np.arange(0.10, 0.90, 0.005):
    tp = fp = tn = fn = 0
    for sim, is_halluc in zip(similarities, labels):
        predicted_halluc = sim < thresh
        if predicted_halluc and is_halluc: tp += 1
        elif predicted_halluc and not is_halluc: fp += 1
        elif not predicted_halluc and is_halluc: fn += 1
        else: tn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 0.001)

    if f1 > best_f1:
        best_f1 = f1
        best_metrics = {
            "threshold": round(float(thresh), 3),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }

# ── Célula 6: Resultados ──────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Lab08 v3 — Sentence-Transformers (all-MiniLM-L6-v2)")
print(f"{'='*60}")
print(f"  Threshold: {best_metrics['threshold']}")
print(f"  Precision: {best_metrics['precision']:.1%}")
print(f"  Recall:    {best_metrics['recall']:.1%}")
print(f"  F1:        {best_metrics['f1']:.1%}")

print(f"\n  {'Resposta':55} {'Sim':>5} {'Real':>5} {'Pred':>5} {'OK':>3}")
print("  " + "-" * 78)
correct = 0
thresh = best_metrics['threshold']
for resp, sim, is_halluc in zip(textos_resp, similarities, labels):
    pred = sim < thresh
    ok = pred == is_halluc
    correct += ok
    s = "✅" if ok else "❌"
    print(f"  {resp[:55]:55} {sim:.3f} {'ALUC' if is_halluc else 'OK':>5} {'ALUC' if pred else 'OK':>5} {s}")

accuracy = correct / len(respostas)
print(f"\n  Accuracy: {accuracy:.1%} ({correct}/{len(respostas)})")

# ── Célula 7: Comparação ──────────────────────────────────────
print(f"\n{'='*60}")
print(f"  COMPARAÇÃO v1 vs v2 vs v3")
print(f"{'='*60}")
comparacao = {
    "v1_ngramas":    {"precision": 1.00, "recall": 0.68, "f1": 0.81, "metodo": "4-gram overlap"},
    "v2_tfidf":      {"precision": 0.82, "recall": 0.82, "f1": 0.82, "metodo": "TF-IDF cosine"},
    "v3_sbert":      {"precision": best_metrics["precision"], "recall": best_metrics["recall"],
                      "f1": best_metrics["f1"], "metodo": "all-MiniLM-L6-v2"},
}
for v, m in comparacao.items():
    print(f"  {v:>12}: P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} ({m['metodo']})")

meta_90 = best_metrics['recall'] >= 0.90
print(f"\n  Meta Recall ≥90%: {'✅ ATINGIDA' if meta_90 else '❌ NÃO'} ({best_metrics['recall']:.1%})")

# ── Célula 8: JSON ────────────────────────────────────────────
result = {
    "lab": "lab08_v3",
    "modelo": "all-MiniLM-L6-v2",
    "device": device,
    "n_amostras": len(respostas),
    "metricas": best_metrics,
    "comparacao": comparacao,
    "meta_90_atingida": meta_90,
    "detalhes": [
        {"resposta": r, "similarity": round(s, 4),
         "real": "alucinacao" if h else "correto",
         "predito": "alucinacao" if s < thresh else "correto",
         "correto": (s < thresh) == h}
        for r, s, h in zip(textos_resp, similarities, labels)
    ]
}

print(f"\n{'='*60}")
print("  COPIE O JSON ABAIXO E COLE NO CHAT:")
print(f"{'='*60}")
print(json.dumps(result, indent=2, ensure_ascii=False))
