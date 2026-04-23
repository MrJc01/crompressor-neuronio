# PESQUISA2 — Referências Científicas

**Data:** 2026-04-23
**Objetivo:** Papers e recursos que fundamentam a criação do CromGPT

---

## 1. Vector Quantization em Redes Neurais

| Paper | Autores | Ano | Contribuição |
|-------|---------|-----|-------------|
| **VQ-VAE** | van den Oord et al. | 2017 | Primeiro uso de VQ como camada treinável em NNs |
| **VQ-VAE-2** | Razavi et al. | 2019 | Hierarquia de codebooks para imagens de alta resolução |
| **Residual VQ (SoundStream)** | Zeghidour et al. | 2021 | Múltiplos codebooks residuais para áudio |
| **Product Quantization** | Jégou et al. | 2011 | Dividir vetores em sub-vetores e quantizar independentemente |

### 🎯 Relevância para CromGPT
> O VQ-VAE provou que codebooks treináveis funcionam como representação. A CromLinear estende isso: em vez de codebooks em espaço latente, os **pesos da rede** são o codebook.

---

## 2. Straight-Through Estimator & Alternativas

| Paper | Autores | Ano | Contribuição |
|-------|---------|-----|-------------|
| **Estimating Gradients for Discrete Variables** | Bengio et al. | 2013 | STE: tratar quantização como identidade no backward |
| **Gumbel-Softmax** | Jang et al. | 2017 | Amostragem diferenciável de variáveis discretas |
| **REINFORCE** | Williams | 1992 | Gradientes via policy gradient (mais variância) |

### 🎯 Relevância para CromGPT
> STE é a nossa primeira opção. Se gerar gradientes ruidosos demais, Gumbel-Softmax é o fallback.

---

## 3. Quantization-Aware Training (QAT)

| Paper | Autores | Ano | Contribuição |
|-------|---------|-----|-------------|
| **Quantization and Training of NNs** | Jacob et al. (Google) | 2018 | QAT: simular quantização durante treino |
| **Binary Neural Networks** | Courbariaux et al. | 2016 | Pesos binários (-1, +1) |
| **Ternary Weight Networks** | Li et al. | 2016 | Pesos ternários (-1, 0, +1) |
| **GPTQ** | Frantar et al. | 2023 | Quantização pós-treino com segunda ordem |
| **AWQ** | Lin et al. | 2024 | Activation-aware quantização |

### 🎯 Relevância para CromGPT
> QAT prova que treinar com pesos quantizados é viável. CromLinear vai além: em vez de quantizar para INT8, quantizamos para **índices de codebook**.

---

## 4. Arquiteturas de LLM

| Arquitetura | Paper | Ano | Tipo |
|-------------|-------|-----|------|
| **Transformer** | Vaswani et al. | 2017 | Attention + FFN |
| **GPT-2** | Radford et al. | 2019 | Decoder-only Transformer |
| **Mamba** | Gu & Dao | 2023 | State Space Model (sem Attention) |
| **RWKV** | Peng et al. | 2023 | RNN + Transformer hybrid |
| **RetNet** | Sun et al. | 2023 | Retentive Network |
| **Mixture-of-Experts** | Fedus et al. | 2022 | Sparse routing |
| **TinyLlama** | Zhang et al. | 2024 | 1.1B treinado eficientemente |
| **Phi-2** | Microsoft | 2023 | 2.7B com dados curados |

### 🎯 Relevância para CromGPT
> Começamos com GPT-2 style (Transformer decoder-only). Se funcionar, testamos Mamba como alternativa leve.

---

## 5. Datasets em Português

| Dataset | Fonte | Tamanho | Tipo |
|---------|-------|---------|------|
| **brWac** | UFRGS | ~2.7B tokens | Web crawl PT-BR filtrado |
| **Carolina** | USP/Caravelas | ~1B tokens | Corpus de referência PT-BR |
| **CulturaX-pt** | HuggingFace | ~30B tokens | mC4 + OSCAR filtrado |
| **Wikipedia-PT** | Wikimedia | ~200M tokens | Enciclopédia |
| **Alpaca-PT** | Comunidade | ~52K instruções | Tradução do Alpaca |
| **Dolly-PT** | Comunidade | ~15K instruções | Tradução do Dolly |

### 🎯 Relevância para CromGPT
> Mix: CulturaX-pt (volume) + Wikipedia-PT (qualidade) + Alpaca-PT (instruções).

---

## 6. Pipelines de Dados (Estado da Arte)

| Pipeline | Organização | Método |
|----------|-------------|--------|
| **The Pile** | EleutherAI | 22 fontes diversas, deduplicação MinHash |
| **RedPajama** | Together AI | Reprodução aberta do dataset do LLaMA |
| **FineWeb** | HuggingFace | 15T tokens, filtros de qualidade agressivos |
| **Dolma** | AI2 | Pipeline transparente com código aberto |

### 🎯 Relevância para CromGPT
> Inspirar nosso pipeline de limpeza: download → dedup → filtro qualidade → tokenização.

---

## Links Úteis

- [HuggingFace Datasets PT](https://huggingface.co/datasets?language=pt)
- [Papers With Code: Quantization](https://paperswithcode.com/task/quantization)
- [train-llm-from-scratch (GitHub)](https://github.com/FareedKhan-dev/train-llm-from-scratch)
- [mini-llm (GitHub)](https://github.com/paulocoutinhox/mini-llm)
- [pierreguillou/gpt2-small-portuguese](https://huggingface.co/pierreguillou/gpt2-small-portuguese)
