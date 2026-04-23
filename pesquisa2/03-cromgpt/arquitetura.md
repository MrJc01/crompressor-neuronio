# Arquitetura CromGPT — Especificação Técnica

---

## Visão Geral

CromGPT = GPT-2 style Transformer onde cada `nn.Linear` é substituída por `CromLinear`.

```
┌─────────────────────────────────────────────────┐
│  Token Embedding (nn.Embedding, vocab → dim)     │
│  + Position Embedding (nn.Embedding, max_len → dim)│
│                                                   │
│  ┌─ Block 0 ──────────────────────────────────┐  │
│  │ LayerNorm → Multi-Head Attention (CromLinear) │  │
│  │ + Residual                                     │  │
│  │ LayerNorm → FFN (CromLinear)                   │  │
│  │ + Residual                                     │  │
│  └────────────────────────────────────────────┘  │
│                                                   │
│  ┌─ Block 1..N ──────────────────────────────┐   │
│  │ (mesmo padrão)                               │   │
│  └────────────────────────────────────────────┘  │
│                                                   │
│  LayerNorm Final                                  │
│  LM Head (Linear ou CromLinear, dim → vocab)     │
└─────────────────────────────────────────────────┘
```

---

## Configuração Base

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| `n_layers` | 12 | Padrão GPT-2 small |
| `n_heads` | 12 | Padrão GPT-2 small |
| `d_model` | 768 | Padrão GPT-2 small |
| `d_ff` | 3072 | 4 × d_model (padrão) |
| `vocab_size` | ~50K | Tokenizador PT |
| `max_seq_len` | 512 | Limitado pelo Colab |
| `K` (codebook) | 256 | Equilíbrio compressão/qualidade |
| `D` (dim centróide) | 64 | Divisor natural de 768 |
| `dropout` | 0.1 | Padrão |

---

## Camadas CromLinear no Modelo

Cada bloco Transformer tem **6 matrizes lineares**:

| Camada | Shape | Substituição |
|--------|-------|-------------|
| Q projection | 768 → 768 | `CromLinear(768, 768, K=256, D=64)` |
| K projection | 768 → 768 | `CromLinear(768, 768, K=256, D=64)` |
| V projection | 768 → 768 | `CromLinear(768, 768, K=256, D=64)` |
| O projection | 768 → 768 | `CromLinear(768, 768, K=256, D=64)` |
| FFN up | 768 → 3072 | `CromLinear(768, 3072, K=256, D=64)` |
| FFN down | 3072 → 768 | `CromLinear(3072, 768, K=256, D=64)` |

**Total por bloco:** 6 CromLinear
**Total no modelo:** 12 × 6 = **72 CromLinear** + LM Head

---

## O que NÃO é CromLinear

| Componente | Tipo | Motivo |
|-----------|------|--------|
| Token Embedding | `nn.Embedding` | Lookup table (já é "quantizada" por natureza) |
| Position Embedding | `nn.Embedding` | Idem |
| LayerNorm | `nn.LayerNorm` | Poucos parâmetros (2 × 768 = 1536 floats) |
| LM Head | TBD | Pode ser tied com Token Embedding |

---

## Estimativa de Parâmetros

### Baseline (nn.Linear)

| Componente | Parâmetros |
|-----------|------------|
| Embeddings | 50K × 768 = 38.4M |
| 72 × Linear(768,768) | 72 × 590K = 42.5M |
| 12 × Linear(768,3072) | 12 × 2.36M = 28.3M |
| 12 × Linear(3072,768) | 12 × 2.36M = 28.3M |
| LayerNorms | ~37K |
| **TOTAL** | **~137M** |

### CromGPT (CromLinear)

| Componente | Parâmetros |
|-----------|------------|
| Embeddings | 38.4M (mesmo) |
| 72 codebooks (K=256, D=64) | 72 × 16K = 1.2M |
| 72 × índices | 72 × 9K = 650K (uint8, não treinável) |
| 12 × FFN codebooks (K=256, D=64) | 24 × 48K = 1.2M |
| LayerNorms | ~37K |
| **TOTAL treináveis** | **~41M** (70% menos!) |

> ⚡ O embedding domina. Os pesos lineares que eram 99M viraram ~2.4M de codebooks.

---

## Estratégia Gradual (Se não convergir)

```
Nível 1: Full CromLinear (todos os 72 Linears → CromLinear)
    ↓ se falhar
Nível 2: Hybrid (FFN = CromLinear, Attention = nn.Linear)
    ↓ se falhar  
Nível 3: Partial (apenas FFN down_proj = CromLinear)
    ↓ se falhar
Nível 4: Caminho A (Fine-tune modelo existente + .crom pós-treino)
```
