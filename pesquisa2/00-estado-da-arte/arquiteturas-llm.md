# Arquiteturas de LLM — Análise para Compatibilidade com .crom

*Pesquisa realizada em 23/04/2026*

---

## 1. Transformer Decoder-Only (GPT-2 Style)

**Paper:** Vaswani et al. (2017) "Attention is All You Need" + Radford et al. (2019) GPT-2
**Tipo:** Attention (Q,K,V) + Feed-Forward Network

### Como funciona
```
Input → Token Embedding → [N × (LayerNorm → Multi-Head Attention → LayerNorm → FFN)] → LM Head → Output
```

Cada camada tem 4 matrizes lineares na Attention (Q, K, V, O) e 2 no FFN (up, down). Todas são `nn.Linear` — multiplicação de matrizes Float32.

### Compatibilidade com CromLinear

| Aspecto | Avaliação |
|---------|-----------|
| Substituir `nn.Linear`? | ✅ Direta — cada Linear vira CromLinear |
| Risco de convergência | ⚠️ Médio — Attention depende de precisão de Q,K,V |
| Ecosystem/tooling | ✅ Excelente — mais estudado do mundo |
| Baseline disponível | ✅ GPT-2, GPT-Neo, etc. |

### Veredicto: ✅ PRIMEIRA OPÇÃO
Melhor para isolar variáveis: se não funcionar, sabemos que o problema é o CromLinear, não a arquitetura.

---

## 2. Mamba (State Space Model — SSM)

**Paper:** Gu & Dao (2023) "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
**Tipo:** RNN-like com estado fixo, sem Attention

### Como funciona
```
Input → Embedding → [N × (Norm → SSM Block → Norm → FFN)] → LM Head → Output
```

Não usa Attention. Processa sequências em tempo linear O(n) em vez de quadrático O(n²). Estado fixo = memória constante.

### Comparação com Transformer

| Feature | Transformer | Mamba |
|---------|------------|-------|
| Complexidade | O(n²) | **O(n)** |
| Inferência | Lenta em seqs longas | **5-10x mais rápida** |
| Memória | Cresce quadraticamente | **Fixa** |
| Recall associativo | **Alto** | Baixo |
| Ecossistema | **Maduro** | Emergente |

### Compatibilidade com CromLinear

| Aspecto | Avaliação |
|---------|-----------|
| Substituir Linear? | ⚠️ Parcial — SSM tem matrizes A, B, C, D próprias |
| Risco | 🔴 Alto — dois experimentos novos ao mesmo tempo |
| Vantagem | Modelo mais leve = mais viável no Colab |

### Veredicto: ⏳ SEGUNDA OPÇÃO
Testar DEPOIS de provar CromLinear no Transformer. Se funcionar, Mamba+CromLinear seria extremamente leve.

---

## 3. RWKV

**Paper:** Peng et al. (2023) "RWKV: Reinventing RNNs for the Transformer Era"
**Tipo:** RNN + Transformer híbrido

Combina a eficiência de RNNs com a expressividade de Transformers. Usa "time-decay" em vez de attention. Pode ser treinado como Transformer mas inferência é O(n).

### Compatibilidade com CromLinear

| Aspecto | Avaliação |
|---------|-----------|
| Substituir Linear? | ⚠️ Possível mas inexplorado |
| Risco | 🔴 Alto — arquitetura menos documentada |
| Comunidade | Menor que Transformer/Mamba |

### Veredicto: ❌ NÃO PARA PESQUISA 2
Muito nicho. Pouco tooling. Guardar para pesquisa futura.

---

## 4. Mixture-of-Experts (MoE)

**Paper:** Fedus et al. (2022) "Switch Transformers"
**Tipo:** Transformer com roteamento sparse

Cada input é processado por apenas 1-2 "experts" (FFNs) de muitos. Permite modelos enormes (trilhões de params) com compute eficiente.

### Compatibilidade com CromLinear

| Aspecto | Avaliação |
|---------|-----------|
| Substituir Linear? | ✅ Cada expert é um FFN Linear |
| Risco | ⚠️ Médio — roteamento pode interferir |
| Relevância | Mais útil para modelos >1B params |

### Veredicto: ❌ NÃO PARA PESQUISA 2
Overkill para 125M params. Guardar para quando escalarmos.

---

## 5. RetNet

**Paper:** Sun et al. (2023) "Retentive Network"
**Tipo:** Substituição de Attention por mecanismo de retenção

### Veredicto: ❌ NÃO PARA PESQUISA 2
Pouca adoção. Poucos toolings. Não vale o risco adicional.

---

## Decisão Final

```
┌─────────────────────────────────────────────────┐
│  PESQUISA 2: Transformer Decoder-Only (GPT-2)   │
│  • Mais estudado = melhor para isolar variáveis  │
│  • Baseline abundante (GPT-2, GPT-Neo)           │
│  • Cada nn.Linear → CromLinear direto            │
│                                                   │
│  FUTURO: Se funcionar, testar Mamba+CromLinear   │
└─────────────────────────────────────────────────┘
```
