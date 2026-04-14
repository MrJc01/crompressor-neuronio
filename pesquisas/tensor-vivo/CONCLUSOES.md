# 🧬 Tensor-Vivo — Conclusões da Pesquisa

> **Data:** 2026-04-14
> **Pesquisador:** MrJc01
> **Tese:** O Crompressor pode substituir diretamente tensores/pesos de redes neurais

---

## Resumo Executivo

A pesquisa testou 3 hipóteses fundamentais sobre usar o Crompressor como
substituto de tensores em redes neurais. Os resultados são claros:

| Hipótese | Resultado | Veredicto |
|---|---|---|
| H1: CDC hash exato encontra dedup em pesos | **NÃO** (0% dedup) | ❌ Refutada |
| H2: Codebook K-Means preserva accuracy | **SIM** (96.97% com 9.4x compressão) | ✅ Confirmada |
| H3: Treinar APENAS o codebook funciona | **SIM** (98.08%, SUPEROU baseline) | ✅ **Confirmada fortemente** |

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

---

## Resposta à Tese Central

> **"O Crompressor pode substituir diretamente os pesos de tensores?"**

### Resposta: **SIM, com ressalvas.**

**O que funciona:**
- ✅ Representar pesos como índices apontando para um codebook de centróides
- ✅ Treinar apenas o codebook (índices congelados) alcança accuracy equivalente
- ✅ Compressão de até 40.8x no espaço de parâmetros treináveis
- ✅ O codebook é um espaço de aprendizado estável e convergente

**O que NÃO funciona como esperado:**
- ❌ CDC hash exato não encontra dedup em pesos (cada neurônio é único)
- ❌ A codificação DNA Base-4 não foi testada ainda (usamos K-Means puro)
- ⚠️ Testado apenas em MNIST MLP (modelo pequeno) — precisa validar em modelos maiores

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
1. **Testar em CIFAR-10 CNN** — validar que funciona além de MNIST
2. **Testar em modelo Transformer** — GPT-2 small (124M params)
3. **Comparar formalmente com LoRA** — mesmos params treináveis, mesma task

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

```
235,146 parâmetros originais
  5,770 parâmetros no codebook (K=128, B=16)
  40.8x compressão
 97.56% accuracy (vs 97.53% baseline)
  0.03% gap — praticamente idêntico
```

> **O neurônio que comprime é o neurônio que pensa.**
> **E o codebook é a memória comprimida desse pensamento.**
