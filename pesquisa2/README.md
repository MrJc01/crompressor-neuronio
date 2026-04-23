<div align="center">

# 🧠 Pesquisa 2 — CromGPT: LLM Nativo com Arquitetura .crom

**"Não comprimimos os pesos depois. Os pesos já nascem comprimidos."**

*Pesquisa original que substitui tensores Float32 por codebooks .crom como representação nativa de uma rede neural, treinando um modelo de linguagem do zero em Português.*

</div>

---

## 📋 Origem

As pesquisas anteriores provaram que:
- **Pesquisa 0:** Active Inference (minimização de Free Energy) substitui buscas exaustivas em grafos neurais
- **Pesquisa 1:** O formato `.crom` comprime pesos reais de redes neurais (98.76% em GPT-2) e a criptografia Ed25519 blinda redes P2P

A **Pesquisa 2** responde à pergunta: *E se os pesos de uma rede neural nunca fossem Float32? E se fossem codebooks .crom desde o nascimento?*

---

## 🧭 Mapa da Pesquisa

```
pesquisa2/
├── README.md                           # ← Este arquivo
├── PLANEJAMENTO.md                     # Checklist extenso (~200 items)
├── CONCLUSOES.md                       # Veredictos (preenchido ao final)
├── REFERENCIAS.md                      # Papers: VQ em NNs, STE, PQ, etc.
│
├── 00-estado-da-arte/                  📚 Survey de arquiteturas e técnicas
│   ├── README.md
│   ├── arquiteturas-llm.md            Transformer, Mamba, RWKV, MoE, RetNet
│   ├── quantizacao-em-redes.md        VQ, PQ, Binary NNs, QAT, STE
│   └── pipelines-de-dados.md          Como Big Techs filtram/treinam
│
├── 01-data-pipeline/                   📊 Coleta e preparação de dados PT
│   ├── README.md
│   ├── datasets-portugues.md          Catálogo de datasets disponíveis
│   └── pipeline.md                    Fluxo: download → limpeza → tokenização
│
├── 02-crom-layer/                      ⚡ CromLinear: O coração da pesquisa
│   ├── README.md
│   ├── teoria-cromlinear.md           Matemática: forward, backward, STE
│   └── testes-sinteticos.md           Plano de validação isolada
│
├── 03-cromgpt/                         🤖 Modelo completo CromGPT
│   ├── README.md
│   ├── arquitetura.md                 Config: layers, heads, K, D
│   └── treinamento.md                 Loop, hyperparams, Colab
│
├── 04-avaliacao/                       📈 Benchmarks e comparação com baseline
│   ├── README.md
│   └── metricas.md                    Perplexidade, diversidade, tamanho
│
├── 05-formato-crom-v3/                 💾 Serialização nativa de modelos
│   ├── README.md
│   └── especificacao.md               Header + codebooks + índices
│
├── labs/                               🔬 Código de cada experimento
│   ├── lab25-data-pipeline/
│   ├── lab26-crom-linear/
│   ├── lab27-cromgpt-base/
│   ├── lab28-baseline-comparison/
│   ├── lab29-crom-v3-format/
│   └── lab30-instruction-tuning/
│
├── papers/                             📝 Escritos DEPOIS dos resultados
│   ├── papel0.md                      CromLinear: teoria + convergência
│   └── papel1.md                      CromGPT: treinamento + avaliação
│
├── resultados/                         📊 JSONs de cada lab
│
├── diagramas/                          📐 Visualizações da arquitetura
│
└── notebooks/
    └── colab_train.ipynb              ☁️ Notebook para Google Colab
```

---

## 🔬 Os 6 Eixos de Pesquisa

| # | Eixo | Labs | Pergunta Central |
|:--|:-----|:-----|:-----------------|
| 0 | **Estado da Arte** | — | O que já existe em VQ/PQ para redes neurais? |
| 1 | **Data Pipeline** | 25 | Como coletar e preparar corpus PT de qualidade? |
| 2 | **CromLinear** | 26 | Um neurônio com pesos-codebook converge? |
| 3 | **CromGPT** | 27 | Um Transformer inteiro com CromLinear funciona? |
| 4 | **Avaliação** | 28 | CromGPT gera texto coerente em PT? |
| 5 | **Formato .crom v3** | 29 | O modelo inteiro cabe num .crom nativo? |

---

## 🔗 Conexões com Pesquisas Anteriores

| Pesquisa | O que provamos | Base para Pesquisa2 |
|:---------|:---------------|:--------------------|
| [pesquisa0](../pesquisa0/CONCLUSOES.md) | Active Inference 12.7x, KV Cache 94.2% | Teoria de Free Energy aplicada ao sampling |
| [pesquisa1](../pesquisa1/papers/papel0.md) | Go nativo <1μs, VQ 98.76%, Ed25519 real | Formato .crom v2, pipeline de compressão |
| [pesquisa1 v3](../pesquisa1/exemplos/) | Dados 100% reais, Nucleus Sampling | Metodologia honesta de avaliação |

---

## ⚠️ Riscos Conhecidos

1. **CromLinear pode não convergir:** Straight-Through Estimator é uma aproximação — gradientes podem ser ruidosos demais
2. **Modelo pequeno (125M) pode ser burro:** GPT-2 small já gera texto medíocre com pesos normais, com codebooks pode ser pior
3. **Colab grátis pode não bastar:** Sessões de 12h podem não ser suficientes para treinar 125M params
4. **Protocolo de mitigação:** Documentar TUDO. Se falhar, cair para Caminho A (fine-tune + .crom pós-treino)

---

<div align="center">

*"Os pesos de uma rede neural são apenas vetores num dicionário. E se o dicionário fosse o modelo?"*

**Pesquisa 2** · Iniciada em Abril de 2026

</div>
