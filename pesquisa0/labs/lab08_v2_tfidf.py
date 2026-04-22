#!/usr/bin/env python3
"""
Lab08 v2 — Detector de Alucinação com Embeddings (TF-IDF + Cosine)
Melhoria sobre Lab08 v1 (n-gramas): usa representações vetoriais para capturar
similaridade semântica, não apenas overlap léxico.

Objetivo: Recall > 90% (Lab08 v1 atingiu apenas 68%)
"""
import json, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = os.path.dirname(__file__)
RESULTADOS = os.path.join(BASE, '..', 'resultados')

# ── Dataset de teste ─────────────────────────────────────────
# Contexto: informação real que o modelo deveria usar
contexto = """
A Torre Eiffel tem 330 metros de altura e fica em Paris, França.
Foi construída em 1889 por Gustave Eiffel para a Exposição Universal.
Recebe cerca de 7 milhões de visitantes por ano.
É feita de ferro puddled e pesa aproximadamente 7.300 toneladas.
A torre possui 3 andares acessíveis ao público.
"""

# Respostas a avaliar: mix de corretas e alucinações
respostas = [
    # Corretas (ground truth)
    ("A Torre Eiffel tem 330 metros de altura.", False),
    ("Ela fica em Paris, na França.", False),
    ("Foi construída em 1889 por Gustave Eiffel.", False),
    ("Recebe cerca de 7 milhões de visitantes.", False),
    ("É feita de ferro e pesa 7300 toneladas.", False),
    # Alucinações (informação inventada)
    ("A Torre Eiffel tem 500 metros de altura.", True),
    ("Foi construída em 1920 por Nikola Tesla.", True),
    ("A torre é feita de aço inoxidável e titânio.", True),
    ("Recebe 50 milhões de visitantes por dia.", True),
    ("A torre possui 15 andares e um heliporto.", True),
    # Sutis (parcialmente corretas)
    ("A Torre Eiffel tem cerca de 300 metros.", False),  # Aproximado OK
    ("Foi construída para a Exposição de 1889.", False),
    ("A torre pesa cerca de 10.000 toneladas.", True),   # Inflacionado
    ("Fica em Lyon, na França.", True),                   # Cidade errada
    ("Foi projetada por Alexandre Eiffel.", True),        # Nome errado
    # Genéricas (não-verificáveis mas plausíveis)
    ("A Torre Eiffel é um monumento icônico.", False),
    ("É uma das atrações mais visitadas do mundo.", False),
    ("A torre foi renovada recentemente.", True),         # Sem base no contexto
    ("O restaurante da torre tem estrela Michelin.", True),
    ("A estrutura é considerada patrimônio da UNESCO.", True), # Sem base
]

print("=" * 60)
print("  Lab08 v2 — Detector de Alucinação (TF-IDF + Cosine)")
print("=" * 60)

# ── Método 1: TF-IDF + Cosine Similarity ─────────────────────
print("\n▶ Método 1: TF-IDF Cosine Similarity")

# Criar corpus: contexto + cada resposta
textos_resp = [r[0] for r in respostas]
labels = [r[1] for r in respostas]  # True = alucinação

# Vetorizar com TF-IDF
vectorizer = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 3),  # Unigrams + bigrams + trigrams
    min_df=1,
    sublinear_tf=True,
)
tfidf_matrix = vectorizer.fit_transform([contexto] + textos_resp)

# Cosine similarity entre contexto (index 0) e cada resposta
ctx_vector = tfidf_matrix[0:1]
resp_vectors = tfidf_matrix[1:]
similarities = cosine_similarity(ctx_vector, resp_vectors)[0]

# ── Buscar threshold ótimo ────────────────────────────────────
print("\n  Buscando threshold ótimo...")
best_f1 = 0
best_thresh = 0
best_metrics = {}

for thresh in np.arange(0.01, 0.50, 0.01):
    tp = fp = tn = fn = 0
    for sim, is_halluc in zip(similarities, labels):
        predicted_halluc = sim < thresh
        if predicted_halluc and is_halluc:
            tp += 1
        elif predicted_halluc and not is_halluc:
            fp += 1
        elif not predicted_halluc and is_halluc:
            fn += 1
        else:
            tn += 1
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 0.001)
    
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh
        best_metrics = {
            "threshold": round(float(thresh), 2),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }

print(f"\n  Threshold ótimo: {best_thresh:.2f}")
print(f"  Precision: {best_metrics['precision']:.1%}")
print(f"  Recall:    {best_metrics['recall']:.1%}")
print(f"  F1:        {best_metrics['f1']:.1%}")

# ── Detalhes por resposta ─────────────────────────────────────
print(f"\n  {'Resposta':50} {'Sim':>5} {'Real':>6} {'Pred':>6} {'OK':>3}")
print("  " + "-" * 75)
correct = 0
for resp, sim, is_halluc in zip(textos_resp, similarities, labels):
    pred = sim < best_thresh
    ok = pred == is_halluc
    correct += ok
    status = "✅" if ok else "❌"
    print(f"  {resp[:50]:50} {sim:.3f} {'ALUC' if is_halluc else 'OK':>6} {'ALUC' if pred else 'OK':>6} {status}")

accuracy = correct / len(respostas)
print(f"\n  Accuracy geral: {accuracy:.1%} ({correct}/{len(respostas)})")

# ── Comparação v1 vs v2 ──────────────────────────────────────
print("\n" + "=" * 60)
print("  COMPARAÇÃO: Lab08 v1 (n-gramas) vs v2 (TF-IDF)")
print("=" * 60)
comparacao = {
    "v1_ngramas": {"precision": 1.00, "recall": 0.68, "f1": 0.81, "metodo": "4-gram overlap"},
    "v2_tfidf": {
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "f1": best_metrics["f1"],
        "metodo": "TF-IDF (1-3gram) + cosine similarity",
    },
}
for versao, m in comparacao.items():
    print(f"  {versao:>12}: P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} ({m['metodo']})")

recall_melhorou = best_metrics["recall"] > 0.68
print(f"\n  Recall melhorou? {'SIM' if recall_melhorou else 'NÃO'} ({best_metrics['recall']:.1%} vs 68%)")
print(f"  Meta >90%?      {'SIM' if best_metrics['recall'] >= 0.90 else 'NÃO'}")

# ── Salvar resultados ─────────────────────────────────────────
results = {
    "metodo": "TF-IDF + Cosine Similarity",
    "n_amostras": len(respostas),
    "n_alucinacoes": sum(labels),
    "n_corretas": sum(1 for l in labels if not l),
    "threshold_otimo": best_thresh,
    "metricas": best_metrics,
    "comparacao_v1_v2": comparacao,
    "detalhes": [
        {"resposta": r, "similarity": round(float(s), 4),
         "real": "alucinacao" if h else "correto",
         "predito": "alucinacao" if s < best_thresh else "correto"}
        for r, s, h in zip(textos_resp, similarities, labels)
    ],
}

os.makedirs(RESULTADOS, exist_ok=True)
out = os.path.join(RESULTADOS, 'lab08_v2_results.json')
with open(out, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n✅ Salvo em {out}")
