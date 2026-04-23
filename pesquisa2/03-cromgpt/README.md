# Eixo 03 — CromGPT: Modelo Completo

> Transformer decoder-only (GPT-2 style) onde TODAS as camadas lineares são CromLinear.

## Objetivo

Montar, treinar e validar um modelo de linguagem completo usando CromLinear.

## Configuração Alvo

| Parâmetro | Valor |
|-----------|-------|
| Tipo | Decoder-only Transformer |
| Layers | 12 |
| Heads | 12 |
| Dim | 768 |
| Codebook K | 256 (ajustável) |
| Codebook D | 64 (ajustável) |
| Vocab | Definido pelo tokenizador |
| Params | ~125M equivalentes |
| Hardware | Google Colab T4 (15GB VRAM) |

## Arquivos

| Arquivo | Conteúdo |
|---------|----------|
| `arquitetura.md` | Detalhes de cada bloco: Embedding, Attention, FFN, LM Head |
| `treinamento.md` | Loop de treino: optimizer, scheduler, checkpointing |

## Lab Associado

`labs/lab27-cromgpt-base/` — Implementação completa do modelo + script de treino.

## Critério de Conclusão

Loss diminui consistentemente ao longo de múltiplas epochs. Modelo gera texto (mesmo que imperfeito).
