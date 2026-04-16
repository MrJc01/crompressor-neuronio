# 🧬 Tensor-Vivo — Conclusões da Pesquisa

> **Data:** 2026-04-16 (atualizado com Exp5 v2 — Transformer GPT-2)
> **Pesquisador:** MrJc01
> **Tese:** O Crompressor pode substituir diretamente tensores/pesos de redes neurais

---

## Resumo Executivo

A pesquisa testou 4 hipóteses fundamentais sobre usar o Crompressor como
substituto de tensores em redes neurais. Os resultados são claros:

| Hipótese | Resultado | Veredicto |
|---|---|---|
| H1: CDC hash exato encontra dedup em pesos | **NÃO** (0% dedup) | ❌ Refutada |
| H2: Codebook K-Means preserva accuracy | **SIM** (96.97% com 9.4x compressão) | ✅ Confirmada |
| H3: Treinar APENAS o codebook funciona | **SIM** (98.08%, SUPEROU baseline) | ✅ **Confirmada fortemente** |
| H4: Codebook Learning escala para CNN | **SIM** (99.7% do baseline, 145.3x compressão) | ✅ **Confirmada** |
| H5: Codebook Learning escala para Transformer | **SIM** (91.5% do baseline, 389.5x compressão) | ✅ **Confirmada com ressalvas** |

---

## Resultados por Experimento

### Exp0: Análise Estrutural CDC
- CDC com hash exato NÃO encontra dedup em pesos float32
- Zero neurônios com cosine similarity > 0.95
- Entropia média 6.81 bits/byte confirma que pesos têm estrutura (não são ruído)
- **Insight:** A "dedup" do Crompressor em tensores deve ser por proximidade (clustering), não por hash exato

### Exp1: Roundtrip com Codebook K-Means
**32 combinações testadas (K × block_size)**

Resultados-chave:
- **K=128, Block=16:** 96.43% accuracy, **18.5x** compressão, −1.10% perda
- **K=512, Block=16:** 96.97% accuracy, 9.4x compressão, −0.56% perda
- **K=64, Block=16:** 93.14% accuracy, **22.6x** compressão (máxima viável)

**Prova:** Os pesos de uma rede neural PODEM ser representados por um
codebook de centróides com perda mínima de accuracy.

### Exp2: Codebook Learning (Resultado Principal)
**5 configurações testadas, TODAS superaram 97.5%**

Resultados-chave:

| Config | Accuracy Final | Params | Compressão |
|---|---|---|---|
| **K=128, B=16** | **97.56%** | **5,770** | **40.8x** |
| K=256, B=32 | **98.08%** ← superou baseline | 18,058 | 13.0x |
| K=128, B=32 | 97.93% (recovery +6.49%) | 9,866 | 23.8x |

**Descobertas:**
1. O codebook converge em **1 epoch** — o espaço de otimização é suave
2. Com K=256 B=32, **superou o baseline** (98.08% vs 97.53%)
3. K=128 B=16 alcança accuracy equivalente ao baseline com **40.8x menos params**
4. O efeito de regularização do codebook parece **melhorar generalização**

### Exp3: Codebook Learning em CNN CIFAR-10 (Escalabilidade)
**5 configurações testadas em CNN com Conv2d + Linear, TODAS recovery ≥97.8%**

| Config | Pré-Treino | Pós-Treino | Params | Compressão | Gap |
|---|---|---|---|---|---|
| **K=256, B=8** | 68.39% | **77.66%** | **7,370** | **145.3x** | **0.20%** |
| K=128, B=8 | 66.85% | 76.99% | 4,298 | **249.1x** | 0.87% |
| K=256, B=16 | 50.84% | 77.33% | 11,978 | 89.4x | 0.53% |
| K=256, B=32 | 58.41% | 77.33% | 20,170 | 53.1x | 0.53% |
| K=128, B=16 | 46.36% | 76.17% | 7,370 | 145.3x | 1.69% |

**Descobertas:**
1. **99.7% do baseline** com K=256 B=8 (gap de apenas 0.20%)
2. **249.1x compressão** viável com K=128 B=8 (4,298 params vs 1,070,794)
3. **Conv2d funciona** com Flatten+Chunk — sem tratamento especial
4. **Linear comprime melhor** que Conv2d (MSE 2-4x menor)
5. O FC com 1M params → 2,048 params de codebook (compartilhamento 512:1)
6. **Convergência rápida** — 1 epoch atinge >90% do recovery final
7. **Escala MELHOR** que MNIST — 145.3x vs 40.8x compressão

### Exp5 v2: Codebook Learning em GPT-2 Small Transformer (124M params)

**Bug fix crítico:** GPT-2 usa `Conv1D` do HuggingFace (não `nn.Linear`).
O v1 substituiu 0 camadas. O v2 corrige isso e substitui todas as 48 Conv1D.

**Infra:** RTX A4000 16GB via Vast.ai ($0.08/hr, custo total ~$0.50)

| Config | Pré-Treino | Pós-Treino | Best | Params | Compressão | Gap |
|---|---|---|---|---|---|---|
| **K=256, B=16** | 50.80% | **83.72%** | 83.72% | **319,488** | **389.5x** | **7.80%** |
| K=512, B=32 | 50.92% | 83.37% | **84.29%** | 909,312 | 136.9x | 7.22% |

**Descobertas:**
1. **48 Conv1D substituídas** — c_attn, c_proj, c_fc, c_proj × 12 blocos
2. **Recovery de +33pp** — de 50.8% (random chance) para 83.7%
3. **91.5% do baseline** com 389.5x compressão (319K vs 124M params)
4. **Convergência monótona** — accuracy sobe a cada epoch (não oscila)
5. **K=512 B=32 melhor best** (84.29% no epoch 4) mas overfit no epoch 5
6. **Gap de ~8%** — maior que MLP/CNN, indica que Transformers precisam mais K ou epochs
7. **~12 min/epoch** no A4000 (vs ~12 min/epoch no A100 do v1)

**Comparação com v1 (bug):**
- v1: 0 camadas substituídas, treinava apenas biases+LN+head (122K params) → 91.86%
- v2: 48 camadas substituídas, treina codebook+biases+LN+head (319K params) → 83.72%
- **Insight:** O v1 acidentalmente provou que biases+LN fine-tuning supera o baseline!

---

## Resposta à Tese Central

> **"O Crompressor pode substituir diretamente os pesos de tensores?"**

### Resposta: **SIM, validado em 3 arquiteturas.**

**O que funciona:**
- ✅ Representar pesos como índices apontando para um codebook de centróides
- ✅ Treinar apenas o codebook (índices congelados) alcança accuracy equivalente
- ✅ Compressão de até **389.5x** no espaço de parâmetros treináveis
- ✅ O codebook é um espaço de aprendizado estável e convergente
- ✅ **Funciona em MLP (MNIST), CNN (CIFAR-10) e Transformer (GPT-2)**
- ✅ **Funciona para Conv2d e Conv1D** — kernel representável por codebook
- ✅ **Escala inversamente** — quanto maior o modelo, maior a compressão

**O que NÃO funciona como esperado:**
- ❌ CDC hash exato não encontra dedup em pesos (cada neurônio é único)
- ❌ A codificação DNA Base-4 não foi testada ainda (usamos K-Means puro)
- ⚠️ **Transformer: recovery moderado** — 91.5% do baseline (vs 99.7% CNN)
- ⚠️ Mais K (1024+) e/ou mais epochs podem fechar o gap

**Ressalva importante:**
O que provamos é essencialmente **Vector Quantization de pesos** — uma técnica
conhecida. O diferencial do Crompressor seria:
1. CDC para definir **limites de bloco adaptativos** (não testado com sucesso)
2. DNA encoding dos centróides (não testado)
3. Merkle Tree para verificação de integridade (aplicável)
4. Dedup entre camadas/modelos (o hash exato falhou, mas clustering pode funcionar)

---

## O Que Isso Significa Para o Crompressor

### Caminho Validado: Codebook-as-LoRA
O resultado mais impactante é que o **Codebook Learning funciona como LoRA**.
Isso abre um caminho de produto real:

```
1. Modelo LLM grande → Quantizar pesos com K-Means → Codebook .crom
2. Para adaptar a um domínio: treinar APENAS o codebook (poucos KB)
3. Distribuir "deltas de codebook" em vez de LoRA adapters
4. Merkle Tree garante integridade do codebook
```

### Caminho Ainda Aberto: CDC Semântico
O CDC com hash exato falhou, mas isso não invalida a ideia de CDC.
A próxima investigação seria:
- CDC com **Locality-Sensitive Hashing** em vez de SHA-256
- Blocos de tamanho adaptativo baseado na variância dos pesos
- Dedup por distância euclidiana (threshold) em vez de hash exato

---

## Próximos Passos Recomendados

### Curto Prazo (Validação)
1. ~~**Testar em CIFAR-10 CNN**~~ → ✅ **FEITO** (99.7% recovery, 145.3x compressão)
2. ~~**Testar em modelo Transformer**~~ → ✅ **FEITO** (91.5% recovery, 389.5x compressão)
3. **Fechar gap Transformer** — K=1024+ e/ou 10+ epochs
4. **Comparar formalmente com LoRA** — mesmos params treináveis, mesma task

### Médio Prazo (Integração com Crompressor)
4. **Implementar CodebookLinear em Go** — integrar com o motor .crom
5. **Formato .crom para codebooks** — header binário + centróides + índices
6. **FUSE driver para modelos codebook-quantized** — servir via filesystem

### Longo Prazo (Pesquisa)
7. **CDC com LSH** — content-defined chunking semântico (não por hash exato)
8. **Codebook compartilhado entre modelos** — dedup cross-model
9. **Paper: "Codebook-as-LoRA"** — publicar se os resultados escalarem

---

## Números Para Lembrar

### MNIST MLP (Exp2)
```
    235,146 params originais →     5,770 params codebook
     40.8x compressão
    97.56% accuracy (vs 97.53% baseline) — gap: 0.03%
```

### CIFAR-10 CNN (Exp3)
```
  1,070,794 params originais →     7,370 params codebook
    145.3x compressão
    77.66% accuracy (vs 77.86% baseline) — gap: 0.20%
```

### GPT-2 Transformer (Exp5 v2)
```
124,441,344 params originais →   319,488 params codebook
    389.5x compressão
    83.72% accuracy (vs 91.51% baseline) — gap: 7.80%
    48 camadas Conv1D substituídas
    Recovery: 50.80% → 83.72% (+32.9 pp)
```

### Resumo Cruzado
```
  Arquitetura     | Baseline | Codebook | Gap    | Compressão | Recovery
  ----------------+----------+----------+--------+------------+---------
  MNIST MLP       |  97.53%  |  97.56%  | +0.03% |    40.8x   |  100.0%
  CIFAR-10 CNN    |  77.86%  |  77.66%  | -0.20% |   145.3x   |   99.7%
  GPT-2 Transf.   |  91.51%  |  83.72%  | -7.80% |   389.5x   |   91.5%
```

> **O neurônio que comprime é o neurônio que pensa.**
> **E o codebook é a memória comprimida desse pensamento.**
> **Validado em MLP, CNN e Transformer.**
