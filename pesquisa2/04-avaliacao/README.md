# Eixo 04 — Avaliação

> Comparação justa: CromGPT vs Baseline (nn.Linear) no mesmo corpus.

## Objetivo

Medir objetivamente se o CromGPT funciona e quanto perde (ou ganha) em relação ao modelo com pesos tradicionais.

## Métricas

| Métrica | O que mede | Bom se... |
|---------|-----------|-----------|
| Perplexidade | Surpresa do modelo com texto real | Menor é melhor |
| Diversidade Lexical | % de tokens únicos na saída | Maior é melhor |
| Repetições | Tokens repetidos adjacentes | Menor é melhor |
| Tamanho em Disco | Bytes do modelo salvo | Menor é melhor |
| Tokens/segundo | Velocidade de inferência | Maior é melhor |
| Uso de VRAM | Memória durante inferência | Menor é melhor |

## Lab Associado

`labs/lab28-baseline-comparison/` — Scripts de avaliação + comparação lado-a-lado.

## Critério de Conclusão

Relatório completo com todas as métricas em formato JSON + tabela comparativa.
