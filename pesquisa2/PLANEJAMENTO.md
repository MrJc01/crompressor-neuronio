# PLANEJAMENTO — Pesquisa2: CromGPT (LLM Nativo .crom)

> **Objetivo:** Criar um modelo de linguagem treinado do zero onde os pesos são codebooks `.crom` nativos, usando a camada `CromLinear` em vez de `nn.Linear`.
>
> **Data:** 2026-04-23
> **Status:** 🔄 EM PROGRESSO — 40/87 items (Fase 0 ✅, Fase 2 ✅ parcial, Fase 3 arq ✅)
> **Pré-requisito:** pesquisa0 (129/129) + pesquisa1 v3 (dados reais validados)

---

## 🎯 Painel de Especialistas Simulados (Top 5)

| # | Papel | Contribuição |
|:--|:------|:-------------|
| 1 | **Pesquisador de ML/IA (PhD, 15 anos)** | Validação de CromLinear, STE, convergência, arquitetura Transformer |
| 2 | **Eng. de Compressão Neural (Google Brain)** | Product Quantization, VQ-VAE, codebook collapse, treinamento estável |
| 3 | **NLP Engineer Senior (Meta AI)** | Tokenização PT, curadoria de dados, instruction tuning, avaliação |
| 4 | **SRE/MLOps (20 anos)** | Pipeline de treino, Colab, checkpointing, reprodutibilidade |
| 5 | **Matemático (Otimização)** | Straight-Through Estimator, gradientes, convergência teórica |

### Melhores Práticas Consolidadas

1. **Isolar variáveis:** Testar CromLinear sozinha antes de montar o modelo completo
2. **Baseline obrigatório:** Treinar modelo idêntico com nn.Linear para comparação justa
3. **Checkpoints frequentes:** Colab desconecta — salvar a cada epoch
4. **Seeds fixas:** Toda execução reprodutível com `torch.manual_seed(42)`
5. **Métricas quantitativas:** Perplexidade, cosine similarity, tamanho em disco — sem adjetivos

### Armadilhas Identificadas

- ⚠️ **STE pode não funcionar:** O gradiente aproximado pode ser ruidoso demais para K grande
- ⚠️ **Codebook collapse:** Todos os vetores convergem para o mesmo centróide (problema clássico de VQ)
- ⚠️ **Colab timeout:** Sessões grátis cortam após ~12h. Treinos longos exigem checkpointing
- ⚠️ **Dados PT são escassos:** Muito menos dados de qualidade que inglês
- ⚠️ **125M params pode ser insuficiente:** Modelos pequenos são notoriamente piores em tarefas de instrução

---

## Convenções

> - `[x]` = Completo
> - `[/]` = Em progresso
> - `[ ]` = Não iniciado
> - **[P1]** = Prioridade Alta (fundação)
> - **[P2]** = Prioridade Média (expansão)
> - **[P3]** = Prioridade Baixa (fronteira)

---

## ═══════════════════════════════════════════════════════════
## EIXO 00 — ESTADO DA ARTE (Pesquisa Pura)
## ═══════════════════════════════════════════════════════════

### 0.1 — Arquiteturas de LLM

**Objetivo:** Catalogar todas as arquiteturas existentes e avaliar compatibilidade com .crom.

- [x] **0.1.1** [P1] Pesquisar Transformer (Vaswani 2017): Attention + FFN
- [x] **0.1.2** [P1] Pesquisar GPT-2/GPT-3 (Radford 2019): Decoder-only Transformer
- [x] **0.1.3** [P2] Pesquisar Mamba/SSM (Gu 2023): State Space Models, sem Attention
- [x] **0.1.4** [P2] Pesquisar RWKV (Peng 2023): RNN + Transformer hybrid
- [x] **0.1.5** [P2] Pesquisar RetNet (Sun 2023): Retentive Network
- [x] **0.1.6** [P2] Pesquisar Mixture-of-Experts (Fedus 2022): Sparse routing
- [x] **0.1.7** [P1] Documentar prós/contras de cada uma para uso com codebooks
  - **Critério:** ✅ Tabela comparativa em `00-estado-da-arte/arquiteturas-llm.md`

### 0.2 — Quantização em Redes Neurais

**Objetivo:** Entender o estado da arte de pesos quantizados durante treinamento.

- [x] **0.2.1** [P1] Pesquisar Vector Quantization (VQ-VAE, van den Oord 2017)
- [x] **0.2.2** [P1] Pesquisar Straight-Through Estimator (Bengio 2013)
- [x] **0.2.3** [P1] Pesquisar Quantization-Aware Training (QAT, Jacob 2018)
- [x] **0.2.4** [P2] Pesquisar Product Quantization (Jégou 2011)
- [x] **0.2.5** [P2] Pesquisar Binary Neural Networks (Courbariaux 2016)
- [x] **0.2.6** [P2] Pesquisar Residual VQ (Zeghidour 2021)
- [x] **0.2.7** [P2] Pesquisar Gumbel-Softmax (Jang 2017) como alternativa ao STE
- [x] **0.2.8** [P1] Documentar em `00-estado-da-arte/quantizacao-em-redes.md`

### 0.3 — Pipelines de Dados de Big Techs

**Objetivo:** Entender como Google, Meta, Mistral preparam dados.

- [x] **0.3.1** [P1] Pesquisar The Pile (EleutherAI): como foi montado
- [x] **0.3.2** [P1] Pesquisar RedPajama/SlimPajama: filtragem e deduplicação
- [x] **0.3.3** [P1] Pesquisar Dolma (AI2): pipeline de curadoria
- [x] **0.3.4** [P2] Pesquisar FineWeb (HuggingFace 2024): 15T tokens filtrados
- [x] **0.3.5** [P1] Pesquisar datasets PT existentes: brWac, Carolina, CulturaX, mC4-pt
- [x] **0.3.6** [P1] Pesquisar Alpaca/Dolly traduzidos para PT
- [x] **0.3.7** [P1] Documentar em `00-estado-da-arte/pipelines-de-dados.md`

---

## ═══════════════════════════════════════════════════════════
## EIXO 01 — DATA PIPELINE (lab25)
## ═══════════════════════════════════════════════════════════

### 1.1 — Catálogo de Datasets PT

- [ ] **1.1.1** [P1] Listar todos os datasets PT disponíveis no HuggingFace
- [ ] **1.1.2** [P1] Avaliar tamanho, qualidade e licença de cada um
- [ ] **1.1.3** [P1] Selecionar mix final (Wikipedia PT + corpus conversacional + instrução)
- [ ] **1.1.4** [P1] Documentar decisão em `01-data-pipeline/datasets-portugues.md`

### 1.2 — Pipeline de Limpeza

- [ ] **1.2.1** [P1] Implementar download automatizado dos datasets selecionados
- [ ] **1.2.2** [P1] Implementar deduplicação (MinHash ou exact match)
- [ ] **1.2.3** [P1] Implementar filtro de qualidade (comprimento mínimo, detecção de idioma)
- [ ] **1.2.4** [P2] Implementar remoção de conteúdo tóxico/spam
- [ ] **1.2.5** [P1] Gerar estatísticas: total de tokens, distribuição de comprimento
  - **Critério:** Pipeline roda de ponta a ponta e produz corpus limpo

### 1.3 — Tokenizador

- [ ] **1.3.1** [P1] Avaliar tokenizadores PT existentes (pierreguillou, neuralmind/bertimbau)
- [ ] **1.3.2** [P2] Avaliar se vale treinar BPE próprio no nosso corpus
- [ ] **1.3.3** [P1] Selecionar tokenizador final e documentar decisão
- [ ] **1.3.4** [P1] Gerar corpus tokenizado pronto para treinamento
  - **Critério:** Arquivo tokenizado + vocab_size documentado

---

## ═══════════════════════════════════════════════════════════
## EIXO 02 — CAMADA CROMLINEAR (lab26) ⭐ CORAÇÃO DA PESQUISA
## ═══════════════════════════════════════════════════════════

### 2.1 — Implementação da CromLinear

- [x] **2.1.1** [P1] Definir interface: `CromLinear(in_features, out_features, K, D)`
- [x] **2.1.2** [P1] Implementar codebook como `nn.Parameter` treinável: `C` shape `[K, D]`
- [x] **2.1.3** [P1] Implementar índices como parâmetro: `I` shape `[n_blocks]`
- [x] **2.1.4** [P1] Implementar forward: reconstruir W a partir de `C[I]`, depois `y = x @ W`
- [x] **2.1.5** [P1] Implementar backward com Straight-Through Estimator
- [x] **2.1.6** [P1] Implementar inicialização do codebook (K-Means++ ou random)
- [x] **2.1.7** [P2] Implementar anti-codebook-collapse (commitment loss, EMA update)
  - **Critério:** ✅ `CromLinear` compila, forward+backward rodam. Codebook 100% utilizado.

### 2.2 — Validação em Tarefas Sintéticas

- [⚠️] **2.2.1** [P1] Treinar CromLinear em regressão linear: y = W·x + b
  - **Resultado:** Loss não convergiu para <0.1 (oscila em ~2-3). Achado: CromLinear tem dificuldade com regressão pura.
- [x] **2.2.2** [P1] Treinar CromLinear em XOR (não-linear com ReLU)
  - **Resultado:** ✅ Accuracy 100% (igual baseline)
- [x] **2.2.3** [P1] Treinar MLP de 2 camadas CromLinear em MNIST
  - **Resultado:** ✅ **95.21%** (baseline 98.05%, perda de apenas 2.8%, compressão 11.2x)
- [x] **2.2.4** [P1] Comparar convergência: CromLinear vs nn.Linear nas 3 tarefas
  - **Resultado:** ✅ Documentado em `resultados/lab26_cromlinear.json`

### 2.3 — Análise de Sensibilidade

- [ ] **2.3.1** [P2] Variar K (64, 128, 256, 512, 1024) e medir impacto
- [ ] **2.3.2** [P2] Variar D (16, 32, 64, 128) e medir impacto
- [ ] **2.3.3** [P2] Variar learning rate do codebook vs learning rate dos índices
- [ ] **2.3.4** [P2] Testar Gumbel-Softmax como alternativa ao STE
  - **Critério:** Tabela de resultados em `resultados/lab26_sensitivity.json`

---

## ═══════════════════════════════════════════════════════════
## EIXO 03 — CROMGPT COMPLETO (lab27)
## ═══════════════════════════════════════════════════════════

### 3.1 — Arquitetura

- [x] **3.1.1** [P1] Definir config: layers=12, heads=12, dim=768, vocab_size do tokenizador
- [x] **3.1.2** [P1] Implementar Token Embedding + Positional Embedding (nn.Embedding padrão)
- [x] **3.1.3** [P1] Implementar Multi-Head Attention usando CromLinear para Q, K, V, O
- [x] **3.1.4** [P1] Implementar FFN usando CromLinear para up_proj e down_proj
- [x] **3.1.5** [P1] Implementar LayerNorm + Residual connections
- [x] **3.1.6** [P1] Implementar LM Head (projeção para vocab, weight tying)
- [x] **3.1.7** [P1] Contar parâmetros totais e comparar com GPT-2 equivalente
  - **Resultado:** ✅ 12 CromLinear layers, 100% utilização, 9.6x compressão média. Hybrid mode ok.

### 3.2 — Loop de Treinamento

- [ ] **3.2.1** [P1] Implementar DataLoader para corpus tokenizado
- [ ] **3.2.2** [P1] Implementar loss: CrossEntropyLoss no next-token prediction
- [ ] **3.2.3** [P1] Implementar otimizador: AdamW com weight decay
- [ ] **3.2.4** [P1] Implementar scheduler: warmup linear + cosine decay
- [ ] **3.2.5** [P1] Implementar gradient clipping (max_norm=1.0)
- [ ] **3.2.6** [P1] Implementar checkpointing a cada epoch (Colab pode desconectar)
- [ ] **3.2.7** [P1] Implementar logging: loss, perplexidade, LR, tempo por step
- [ ] **3.2.8** [P2] Implementar mixed precision (FP16) se VRAM for apertada
  - **Critério:** Treino roda pelo menos 1 epoch completo sem crash

### 3.3 — Treinamento Efetivo

- [ ] **3.3.1** [P1] Treinar CromGPT por 1 epoch — verificar se loss diminui
- [ ] **3.3.2** [P1] Se loss não diminuir: ajustar LR, K, D, gradient clipping
- [ ] **3.3.3** [P1] Se divergir: tentar CromLinear só no FFN (Attention com nn.Linear)
- [ ] **3.3.4** [P1] Treinar por N epochs até loss estabilizar
- [ ] **3.3.5** [P1] Salvar modelo final
  - **Critério:** Loss diminui consistentemente ao longo das epochs

### 3.4 — Fallback (Caminho A)

- [ ] **3.4.1** [P3] Se CromGPT não convergir após todas as tentativas:
  - [ ] Fine-tune modelo PT existente (pierreguillou/gpt2-small-portuguese)
  - [ ] Comprimir pesos para .crom pós-treino
  - [ ] Documentar análise de falha detalhada

---

## ═══════════════════════════════════════════════════════════
## EIXO 04 — AVALIAÇÃO (lab28)
## ═══════════════════════════════════════════════════════════

### 4.1 — Baseline

- [ ] **4.1.1** [P1] Treinar modelo IDÊNTICO mas com nn.Linear (mesmo corpus, mesmas epochs)
- [ ] **4.1.2** [P1] Salvar baseline para comparação justa

### 4.2 — Métricas Quantitativas

- [ ] **4.2.1** [P1] Medir perplexidade: CromGPT vs Baseline
- [ ] **4.2.2** [P1] Medir diversidade lexical (como pesquisa1 v3)
- [ ] **4.2.3** [P1] Medir tamanho em disco: CromGPT .crom vs Baseline .pt
- [ ] **4.2.4** [P1] Medir velocidade de inferência: tokens/segundo
- [ ] **4.2.5** [P2] Medir uso de VRAM durante inferência

### 4.3 — Testes de Qualidade

- [ ] **4.3.1** [P1] Gerar texto livre: 10 prompts em PT, avaliar coerência
- [ ] **4.3.2** [P2] Perguntas factuais: "Qual a capital do Brasil?" (se dados permitirem)
- [ ] **4.3.3** [P2] Instruções: "Resuma este texto" (se instruction-tuning for feito)
- [ ] **4.3.4** [P1] Comparar outputs lado-a-lado: CromGPT vs Baseline vs GPT-2 PT
  - **Critério:** Relatório completo em `resultados/lab28_evaluation.json`

---

## ═══════════════════════════════════════════════════════════
## EIXO 05 — FORMATO .CROM V3 (lab29)
## ═══════════════════════════════════════════════════════════

### 5.1 — Especificação

- [ ] **5.1.1** [P1] Definir header: magic, versão, n_layers, n_heads, dim, K, D, vocab_size
- [ ] **5.1.2** [P1] Definir body: embedding + codebooks de cada camada + índices + LM head
- [ ] **5.1.3** [P1] Implementar `save_cromgpt(model, path)` 
- [ ] **5.1.4** [P1] Implementar `load_cromgpt(path) → model`
- [ ] **5.1.5** [P1] Validar: salvar → carregar → gerar texto → resultado idêntico
- [ ] **5.1.6** [P2] Comparar tamanho: .crom v3 vs .pt vs .safetensors vs .gguf
  - **Critério:** Formato funciona end-to-end

---

## ═══════════════════════════════════════════════════════════
## EIXO 06 — PAPERS (Após execução)
## ═══════════════════════════════════════════════════════════

### 6.1 — Documentação Final

- [ ] **6.1.1** [P1] Escrever `papel0.md`: CromLinear — teoria, implementação, convergência
- [ ] **6.1.2** [P1] Escrever `papel1.md`: CromGPT — treinamento completo, comparação, análise
- [ ] **6.1.3** [P2] Consolidar `CONCLUSOES.md` com veredictos finais
- [ ] **6.1.4** [P2] Atualizar `REFERENCIAS.md` com papers citados
  - **Critério:** Papers contêm APENAS dados verificados dos resultados reais

---

## Resumo de Items

| Eixo | Items | Prioridade |
|:-----|:------|:-----------|
| 00 — Estado da Arte | 22 ✅ | P1/P2 |
| 01 — Data Pipeline | 12 | P1 |
| 02 — CromLinear | 15 | P1 ⭐ |
| 03 — CromGPT | 17 | P1 |
| 04 — Avaliação | 11 | P1 |
| 05 — Formato v3 | 6 | P1 |
| 06 — Papers | 4 | P1 |
| **TOTAL** | **~87 items** | |
