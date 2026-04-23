# Métricas de Avaliação — CromGPT vs Baseline

---

## Métricas Quantitativas

| # | Métrica | Fórmula | Bom se |
|---|---------|---------|--------|
| 1 | **Perplexidade** | exp(CrossEntropyLoss) | Menor é melhor. Baseline GPT-2 PT: ~30-50 |
| 2 | **Diversidade Lexical** | tokens_únicos / total_tokens | >70% (como pesquisa1 v3) |
| 3 | **Repetições** | n-grams repetidos adjacentes | 0 é ideal |
| 4 | **Tamanho Disco** | bytes do modelo salvo | CromGPT deve ser menor |
| 5 | **Tokens/segundo** | throughput de inferência | Maior é melhor |
| 6 | **VRAM** | pico de uso durante inferência | Menor é melhor |

## Testes de Geração

### Prompts de Avaliação (10 fixos, seed=42)
```
1. "O Brasil é um país"
2. "A inteligência artificial pode"
3. "O Rio de Janeiro é famoso por"
4. "A capital do Brasil é"
5. "Para fazer um bolo de chocolate, você precisa"
6. "A história da humanidade começou"
7. "O sistema solar é composto por"
8. "A programação de computadores é"
9. "A música brasileira é conhecida"
10. "O futuro da tecnologia será"
```

### Critérios de Avaliação Manual
- **Coerência:** O texto faz sentido gramaticalmente?
- **Relevância:** A continuação é relevante ao prompt?
- **Factualidade:** Os fatos mencionados são corretos?
- **Fluência:** O texto parece natural em PT-BR?

## Formato de Saída

```json
{
  "model": "cromgpt-125m",
  "perplexity": 45.2,
  "lexical_diversity": 0.78,
  "repetitions": 0,
  "disk_size_mb": 12.5,
  "tokens_per_second": 150,
  "vram_peak_mb": 2048,
  "prompts": [
    {"prompt": "O Brasil...", "output": "...", "coherence": 4, "relevance": 3}
  ]
}
```
