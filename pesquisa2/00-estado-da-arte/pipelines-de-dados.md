# Pipelines de Dados — Como Big Techs Treinam LLMs

*Pesquisa realizada em 23/04/2026*

---

## 1. O Pipeline Padrão de Treinamento de LLM

```
Web Crawl → Filtragem → Deduplicação → Tokenização → Treinamento → Avaliação
```

Todas as grandes empresas seguem este fluxo. A diferença está nos filtros de qualidade.

---

## 2. Pipelines Referência

### The Pile (EleutherAI, 2020)
- **Tamanho:** 825 GB de texto, ~300B tokens
- **Fontes:** 22 fontes diversas (Wikipedia, ArXiv, GitHub, livros, StackExchange, etc.)
- **Filtragem:** Perplexidade de modelo de linguagem como proxy de qualidade
- **Deduplicação:** MinHash (Jaccard similarity > 0.5 → remove)
- **Lição para nós:** Diversidade de fontes > volume puro

### RedPajama / SlimPajama (Together AI, 2023)
- **Tamanho:** 1.2T tokens (RedPajama) → 627B tokens após limpeza (SlimPajama)
- **Filtragem:** Removeu ~50% do corpus original por baixa qualidade
- **Deduplicação:** MinHash + exact dedup (n-gram)
- **Lição para nós:** Metade dos dados da web é lixo. Filtrar agressivamente.

### FineWeb (HuggingFace, 2024)
- **Tamanho:** 15T tokens
- **Inovação:** Filtros de qualidade treinados com modelo classificador
- **Resultado:** Superou The Pile e RedPajama em benchmarks de downstream
- **Lição para nós:** Classificador de qualidade > regras heurísticas

### Dolma (AI2, 2024)
- **Tamanho:** 3T tokens
- **Inovação:** Pipeline 100% open-source e reprodutível
- **Código:** Disponível no GitHub (OLMo)
- **Lição para nós:** Transparência e reprodutibilidade importam

---

## 3. Datasets em Português — Catálogo

### Datasets de Pré-Treinamento (Volume)

| Dataset | Tamanho | Fonte | Qualidade | HuggingFace |
|---------|---------|-------|-----------|-------------|
| **CulturaX-pt** | ~30B tokens | mC4 + OSCAR filtrado | ⭐⭐⭐ Alta (dedup + filtro) | `uonlp/CulturaX` |
| **brWac** | ~2.7B tokens | Web crawl BR filtrado | ⭐⭐ Média | UFRGS |
| **Carolina** | ~1B tokens | Web diversa BR | ⭐⭐⭐ Alta (USP, curado) | `carolina-c4ai/corpus-carolina` |
| **Wikipedia-PT** | ~200M tokens | Enciclopédia | ⭐⭐⭐⭐ Muito Alta | `wikimedia/wikipedia` |
| **Portuguese-PD** | Grande | Domínio público BR | ⭐⭐ Média | HuggingFace |

### Datasets de Instrução (Fine-tuning)

| Dataset | Tamanho | Tipo | HuggingFace |
|---------|---------|------|-------------|
| **Alpaca-PT** | ~52K pares | Instrução traduzida | Comunidade |
| **Dolly-PT** | ~15K pares | Instrução traduzida | Comunidade |
| **GigaVerbo** | Grande | Conversacional PT | USP/Tucano |

### Modelos PT Existentes (Referência)

| Modelo | Params | Base | Corpus |
|--------|--------|------|--------|
| `pierreguillou/gpt2-small-portuguese` | 125M | GPT-2 | Wikipedia PT |
| `neuralmind/bert-base-portuguese-cased` | 110M | BERT | brWac |
| Tucano (USP) | Vários | LLaMA | GigaVerbo |
| Sabiá (Maritaca AI) | 7B+ | LLaMA | Corpus PT proprietário |

---

## 4. Plano para o CromGPT

### Mix de Dados Proposto

```
┌─────────────────────────────────────────────────────┐
│  CORPUS CROMGPT (Proposta Inicial)                   │
│                                                       │
│  70% Wikipedia-PT (~200M tokens)                     │
│    → Alta qualidade, factual, gramaticalmente correto│
│                                                       │
│  20% Carolina (~subset, ~200M tokens)                │
│    → Diversidade de estilo e vocabulário              │
│                                                       │
│  10% Alpaca-PT (~52K instruções)                     │
│    → Capacidade de seguir instruções                  │
│                                                       │
│  TOTAL ALVO: ~400-500M tokens                        │
│  (viável para 125M params no Colab em ~6-12 horas)   │
└─────────────────────────────────────────────────────┘
```

### Pipeline de Limpeza

```
1. Download: HuggingFace datasets library
2. Filtro de idioma: fasttext lid.176.bin (detectar PT vs outros)
3. Filtro de qualidade: comprimento mínimo (>50 chars), remoção de boilerplate
4. Deduplicação: exact match (hash) + MinHash (fuzzy)
5. Tokenização: tokenizador PT existente (vocab ~32K)
6. Formato final: arquivo tokenizado para DataLoader
```

### Tokenizador

**Decisão:** Usar tokenizador existente do `pierreguillou/gpt2-small-portuguese`
- Já treinado em Wikipedia PT
- Vocab ~50K tokens
- Economiza semanas de trabalho
- Compatível com GPT-2 style architecture

> Se os resultados forem ruins por causa do tokenizador, treinamos BPE próprio na Fase 4.

---

## 5. Estimativa de Recursos

| Recurso | Valor |
|---------|-------|
| Corpus total | ~400-500M tokens |
| Modelo | 125M params |
| Hardware | T4 15GB (Colab free) |
| Batch size | 4-8 (limitado pela VRAM) |
| Sequence length | 512-1024 tokens |
| Epochs | 1-3 (dados suficientes para 1 epoch ser útil) |
| Tempo estimado | 6-12 horas no Colab |
| Checkpointing | A cada 1000 steps |
