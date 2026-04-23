# Datasets em Português — Catálogo Completo

---

## Pré-Treinamento (Volume)

| Dataset | Tokens | Fonte | Qualidade | Link |
|---------|--------|-------|-----------|------|
| **CulturaX-pt** | ~30B | mC4+OSCAR filtrado | ⭐⭐⭐ | `uonlp/CulturaX` |
| **Carolina** | ~1B | Web diversa BR (USP) | ⭐⭐⭐ | `carolina-c4ai/corpus-carolina` |
| **Wikipedia-PT** | ~200M | Enciclopédia | ⭐⭐⭐⭐ | `wikimedia/wikipedia` |
| **brWac** | ~2.7B | Web crawl BR (UFRGS) | ⭐⭐ | UFRGS |
| **Portuguese-PD** | Grande | Domínio público BR | ⭐⭐ | HuggingFace |
| **GigaVerbo** | Grande | Conversacional (Tucano) | ⭐⭐⭐ | USP |

## Instrução (Fine-Tuning)

| Dataset | Pares | Tipo | Link |
|---------|-------|------|------|
| **Alpaca-PT** | ~52K | Instrução traduzida | Comunidade |
| **Dolly-PT** | ~15K | Instrução traduzida | Comunidade |

## Mix Selecionado para CromGPT

```
70% Wikipedia-PT  (~200M tokens) — factual, gramática correta
20% Carolina      (~200M subset) — diversidade vocabular
10% Alpaca-PT     (~52K pares)   — seguir instruções
```

**Total alvo:** ~400-500M tokens (viável em Colab T4 grátis)

## Considerações

- **PT-BR vs PT-PT:** Focar em PT-BR (mais dados disponíveis)
- **Qualidade > Volume:** Para 125M params, dados limpos importam mais que volume
- **Deduplicação:** MinHash obrigatório (CulturaX já é dedup, Wikipedia também)
