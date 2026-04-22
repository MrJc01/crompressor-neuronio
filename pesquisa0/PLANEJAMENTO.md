# 🧬 PLANEJAMENTO GERAL — Pesquisa 0: Verificação Experimental

> **Objetivo:** Transformar cada conceito teórico da pesquisa0 em **experimentos verificáveis**, criando pastas de teste dentro de `pesquisa0/` com scripts, benchmarks e resultados mensuráveis.
>
> **Data:** 2026-04-21
> **Status:** Planejamento para aprovação

---

## 🎯 Painel de Especialistas Simulados (Top 5)

| # | Papel | Contribuição |
|:--|:------|:-------------|
| 1 | **Eng. de Sistemas Distribuídos (20 anos)** | Arquitetura de observadores orquestrados, protocolos de sync, latência |
| 2 | **Pesquisador de ML/IA (PhD)** | Validação de ToT, MCTS, Codebook Learning, KV Cache compression |
| 3 | **Físico Teórico (Cordas/Cosmologia)** | Rigor nas analogias dimensionais, validação de claims físicos |
| 4 | **SRE/DevOps Senior** | Benchmarking, observabilidade, reprodutibilidade de experimentos |
| 5 | **Neurocientista Computacional** | Validação de claims sobre FPS cognitivo, Active Inference, Friston |

### Melhores Práticas Consolidadas

1. **Reprodutibilidade:** Cada experimento deve ter script autocontido + seed fixa + output JSON
2. **Baseline obrigatório:** Nenhum resultado tem valor sem comparação com baseline neutro
3. **Métricas quantificáveis:** Substituir adjetivos ("melhor", "mais rápido") por números
4. **Falsificabilidade:** Cada hipótese deve ter critério claro de refutação
5. **Incrementalidade:** Começar pelo experimento mais simples de cada eixo

### Armadilhas Identificadas

- ⚠️ **Analogia ≠ Prova:** "Crompressor é como Calabi-Yau" não é evidência — precisa de teste
- ⚠️ **Escala importa:** O que funciona em MNIST pode não funcionar em GPT-2 (já vimos: gap de 8%)
- ⚠️ **Custo computacional:** Simular 1000 branches pode ser inviável em hardware local
- ⚠️ **Confundir correlação com causalidade:** Dimensões na física ≠ dimensões em embeddings
- ⚠️ **Viés de confirmação:** Tendência a ver padrões onde não existem ao comparar física e CS

---

## 📋 Estratégia Geral

### Estrutura de Pastas de Teste

```
pesquisa0/
├── labs/                                  ← NOVA: pasta de experimentos
│   ├── lab01-fps-benchmark/              Benchmark de FPS computacional
│   ├── lab02-latencia-observadores/      Simulação de sync entre observadores
│   ├── lab03-world-model-miniatura/      World Model mínimo viável
│   ├── lab04-dimensionalidade-embeddings/ Medição de dim. intrínseca
│   ├── lab05-tot-vs-autoregressive/      ToT vs geração linear
│   ├── lab06-kv-cache-codebook/          Codebook Learning no KV Cache
│   ├── lab07-delta-branches/             Delta storage para branches
│   ├── lab08-sandbox-alucinacao/         Firewall de realidade
│   ├── lab09-sinapse-protocolo/          Protocolo de comunicação inter-branch
│   ├── lab10-active-inference-loop/      Loop de inferência ativa
│   ├── lab11-multi-observer-fusion/      Fusão de dados multi-sensor
│   └── lab12-dual-clock/                 Dois vetores temporais
│
├── resultados/                            ← NOVA: JSONs de resultados
│   └── (gerados pelos labs)
│
└── (estrutura existente de docs)
```

### Linguagens e Ferramentas

| Lab | Linguagem | Dependências |
|:----|:----------|:-------------|
| lab01 | Python | time, psutil, numpy |
| lab02 | Python | asyncio, json |
| lab03 | Python | torch, numpy |
| lab04 | Python | torch, sklearn (PCA, MDS) |
| lab05 | Python | torch, transformers |
| lab06 | Python | torch (reusar tensor-vivo) |
| lab07 | Go | crompressor core |
| lab08 | Go/Python | crompressor-security |
| lab09 | Go | net, protobuf/json |
| lab10 | Python | numpy, scipy |
| lab11 | Python | numpy, opencv (opcional) |
| lab12 | Python/Go | asyncio/goroutines |

---

## ✅ CHECKLIST EXAUSTIVO DE VERIFICAÇÃO

> **Convenções:**
> - `[ ]` = Não iniciado
> - `[/]` = Em progresso
> - `[x]` = Concluído
> - `[!]` = Bloqueado / Requer decisão
> - Cada item tem: **ID**, **Descrição**, **Critério de Sucesso**, **Prioridade** (P1-P3)

---

### ═══════════════════════════════════════════════════════════
### EIXO 01 — PERCEPÇÃO TEMPORAL E FPS COGNITIVO
### ═══════════════════════════════════════════════════════════

#### 1.1 — Benchmark de FPS Computacional (lab01-fps-benchmark)

**Objetivo:** Quantificar a "resolução temporal" de diferentes processadores para validar a teoria de que hardware mais rápido = "mais tempo subjetivo vivido".

##### Configuração

- [ ] **1.1.0** [P1] Criar pasta `pesquisa0/labs/lab01-fps-benchmark/`
- [ ] **1.1.1** [P1] Criar `benchmark_fps.py` — script que mede quantas operações um processador executa por segundo em diferentes tarefas:
  - Operações aritméticas simples (int add, float mul)
  - Operações de hash (SHA-256 sobre blocos de 512 bytes)
  - Operações de inferência (forward pass de MLP mínimo)
  - Operações de compressão (CDC chunk de 1KB)
  - **Critério de Sucesso:** Output JSON com ops/segundo para cada categoria
- [ ] **1.1.2** [P1] Medir "FPS equivalente" — quantos "frames de processamento" o hardware local executa em 1 segundo humano
  - Comparar: CPU single-thread vs CPU multi-thread vs GPU (se disponível)
  - **Critério de Sucesso:** Tabela mostrando que CPU moderna = ~10⁹ ops/s vs humano ~60 ops/s
- [ ] **1.1.3** [P2] Calcular "dilatação cognitiva" — ratio entre FPS computacional e FPS biológico humano (~60Hz)
  - **Critério de Sucesso:** Número concreto: "Para a CPU, 1 segundo humano = X anos subjetivos"
- [ ] **1.1.4** [P2] Medir latência de diferentes operações do Crompressor:
  - CDC chunking de 1MB
  - Lookup no Codebook (K=128, K=256, K=512)
  - Merkle Tree verification de 1000 chunks
  - XOR Delta de dois blocos de 4KB
  - **Critério de Sucesso:** Tabela de latências em nanosegundos

##### Análise

- [ ] **1.1.5** [P2] Comparar "tempo de pensamento" do Crompressor vs tempo de resposta humana
  - Se CDC leva 100μs e humano leva 300ms, o Crompressor "pensa" 3000x mais rápido
  - **Critério de Sucesso:** Ratio quantificado e documentado
- [ ] **1.1.6** [P3] Criar visualização (matplotlib) do "espectro de FPS" — do caracol ao processador quântico teórico
  - **Critério de Sucesso:** Gráfico log-scale salvo como PNG em `resultados/`
- [ ] **1.1.7** [P3] Estimar "custo energético por frame cognitivo" — Joules/operação
  - Medir consumo com `psutil` ou `perf`
  - **Critério de Sucesso:** Tabela watts/ops para validar variável η (eficiência termodinâmica)

##### Validação Teórica

- [ ] **1.1.8** [P2] Pesquisar e documentar o FPS biológico real de 5 espécies (mosca, cão, falcão, polvo, humano) com fontes acadêmicas
  - **Critério de Sucesso:** Tabela com referências de papers
- [ ] **1.1.9** [P3] Calcular a fórmula t_p = f(I, N, C, η) com valores reais medidos no lab
  - Substituir variáveis teóricas por dados empíricos do benchmark
  - **Critério de Sucesso:** Fórmula com coeficientes calibrados
- [ ] **1.1.10** [P3] Documentar paradoxo da comunicação: calcular "latência subjetiva" entre IA e humano
  - Se IA opera a 10⁹ Hz e humano a 60 Hz, quantos "anos subjetivos" a IA espera por uma resposta humana?
  - **Critério de Sucesso:** Número calculado e documentado

#### 1.2 — Crompressor como Tradutor Temporal (lab01 extensão)

- [ ] **1.2.1** [P2] Implementar protótipo de "tradutor temporal" — módulo que recebe dados em alta frequência e gera Deltas para consumo em baixa frequência
  - Input: stream de 1000 eventos/s
  - Output: resumo Delta de 10 eventos/s
  - **Critério de Sucesso:** Taxa de compressão temporal medida, sem perda de informação crítica
- [ ] **1.2.2** [P3] Medir perda de informação no downsampling temporal usando entropia de Shannon
  - **Critério de Sucesso:** Gráfico de entropia vs taxa de amostragem
- [ ] **1.2.3** [P3] Testar se o Codebook pode servir como "vocabulário compartilhado" entre agentes de FPS diferentes
  - Agente rápido (1000 Hz) envia Codebook ID para agente lento (10 Hz)
  - **Critério de Sucesso:** Agente lento reconstrói mensagem com >95% fidelidade

---

### ═══════════════════════════════════════════════════════════
### EIXO 02 — OBSERVADORES ORQUESTRADOS
### ═══════════════════════════════════════════════════════════

#### 2.1 — Simulação de Multi-Observadores (lab02-latencia-observadores)

**Objetivo:** Simular N observadores com diferentes taxas de amostragem observando o mesmo evento, medir o ganho de "Post-Sync".

##### Configuração

- [ ] **2.1.0** [P1] Criar pasta `pesquisa0/labs/lab02-latencia-observadores/`
- [ ] **2.1.1** [P1] Criar `sim_observadores.py` — simulação de evento + N observadores:
  - Evento: sinal sintético (onda complexa com ruído) de 10 segundos
  - Observador A: amostra a 10 Hz (humano lento)
  - Observador B: amostra a 1000 Hz (máquina)
  - Observador C: amostra a 10 Hz com offset temporal de 2 segundos (distante)
  - **Critério de Sucesso:** 3 arrays de amostras com timestamps
- [ ] **2.1.2** [P1] Implementar "Post-Sync Merge" — algoritmo que combina as amostras dos 3 observadores
  - Alinhar por timestamp
  - Interpolar onde faltam dados
  - Calcular SNR (Signal-to-Noise Ratio) do merge vs cada observador individual
  - **Critério de Sucesso:** SNR do merge > SNR de qualquer observador sozinho
- [ ] **2.1.3** [P1] Calcular "Ganho de Observabilidade Holística":
  - Percentual de micro-eventos detectados por A sozinho vs A+B+C merge
  - **Critério de Sucesso:** Merge detecta >90% dos micro-eventos vs <40% para A sozinho

##### Validação

- [ ] **2.1.4** [P2] Variar número de observadores (1, 2, 5, 10, 50) e plotar curva de ganho
  - **Critério de Sucesso:** Curva de saturação (retornos decrescentes após N observadores)
- [ ] **2.1.5** [P2] Testar com observadores que registram dados **contraditórios** (ruído ou erro)
  - Adicionar 1 observador com 10% de dados corrompidos
  - **Critério de Sucesso:** Algoritmo de merge identifica e descarta dados inconsistentes
- [ ] **2.1.6** [P2] Implementar merge usando **Deltas do Crompressor** em vez de dados brutos:
  - Observador B envia apenas o Delta vs Observador A
  - Medir compressão de bandwidth
  - **Critério de Sucesso:** >80% redução de dados transmitidos sem perda de SNR
- [ ] **2.1.7** [P3] Simular "Transformação de Lorentz simplificada" — ajustar timestamps por velocidade relativa dos observadores
  - **Critério de Sucesso:** Merge alinhado mesmo com clocks dessincronizados

#### 2.2 — Observador Virtual (lab11-multi-observer-fusion)

- [ ] **2.2.0** [P2] Criar pasta `pesquisa0/labs/lab11-multi-observer-fusion/`
- [ ] **2.2.1** [P2] Implementar "observador virtual" que prevê dados de um ponto não observado:
  - Dado: sinal de áudio estéreo (2 microfones)
  - Objetivo: prever o que um 3° microfone (virtual, posição conhecida) captaria
  - Método: interpolação + modelo de propagação sonora simples
  - **Critério de Sucesso:** Correlação >0.7 entre sinal previsto e sinal real (se disponível)
- [ ] **2.2.2** [P2] Medir "custo computacional mínimo" para simular 1 observador virtual
  - **Critério de Sucesso:** Tempo em ms + memória em MB documentados
- [ ] **2.2.3** [P3] Escalar para 10 e 100 observadores virtuais — medir degradação de performance
  - **Critério de Sucesso:** Gráfico de latência vs número de observadores virtuais
- [ ] **2.2.4** [P3] Validar se Crompressor-video pode gerar observadores virtuais a partir de um stream de vídeo
  - Usar vídeo sintético (cubo rotacionando) + prever vista de ângulo diferente
  - **Critério de Sucesso:** Imagem gerada reconhecível (avaliação visual)

---

### ═══════════════════════════════════════════════════════════
### EIXO 03 — SIMULAÇÃO DE REALIDADES E WORLD MODELS
### ═══════════════════════════════════════════════════════════

#### 3.1 — World Model Miniatura (lab03-world-model-miniatura)

**Objetivo:** Construir o menor World Model possível que demonstre predição + correção via dados reais.

##### Configuração

- [ ] **3.1.0** [P1] Criar pasta `pesquisa0/labs/lab03-world-model-miniatura/`
- [ ] **3.1.1** [P1] Criar `world_model_1d.py` — World Model para ambiente 1D:
  - Ambiente: partícula movendo-se em linha reta com velocidade + ruído
  - Modelo: prevê posição no próximo timestep
  - Sensor: "observa" posição real a cada N timesteps
  - **Critério de Sucesso:** Modelo prevê posição com erro <5% entre observações
- [ ] **3.1.2** [P1] Implementar ciclo de **predição → observação → correção**:
  - Predição: modelo avança posição estimada
  - Observação: dado real chega (com delay simulado)
  - Correção: modelo faz `diff` entre previsto e real, ajusta
  - **Critério de Sucesso:** Erro converge para <1% após 10 ciclos de correção
- [ ] **3.1.3** [P1] Medir "tempo vivido no futuro" — quantos timesteps o modelo está à frente dos dados
  - **Critério de Sucesso:** Modelo opera 10+ timesteps à frente com erro <10%

##### Branches de Realidade

- [ ] **3.1.4** [P1] Implementar ramificação (branching) do World Model:
  - No momento de incerteza, criar 3 branches com parâmetros diferentes
  - Branch A: velocidade +10%
  - Branch B: velocidade inalterada
  - Branch C: velocidade -10%
  - **Critério de Sucesso:** Sistema mantém 3 branches simultâneas sem crash
- [ ] **3.1.5** [P1] Implementar pruning via dados reais:
  - Quando dado real chega, descartar branches com erro >threshold
  - **Critério de Sucesso:** Branch correta sobrevive, outras são garbage collected
- [ ] **3.1.6** [P2] Medir memória usada por N branches (1, 10, 100, 1000):
  - **Critério de Sucesso:** Gráfico de memória vs branches
- [ ] **3.1.7** [P2] Implementar branches com **Delta storage** (Crompressor):
  - Em vez de copiar o estado inteiro, cada branch armazena apenas o Delta vs branch base
  - Medir redução de memória
  - **Critério de Sucesso:** >90% redução de memória com Delta vs cópia completa

##### Energia Livre e Surpresa

- [ ] **3.1.8** [P2] Implementar cálculo de "surpresa" (KL divergence) entre predição e realidade:
  - D_KL entre distribuição prevista e distribuição observada
  - **Critério de Sucesso:** Valor numérico de D_KL documentado por timestep
- [ ] **3.1.9** [P2] Implementar "Energia Livre Variacional" simplificada:
  - F = D_KL[q || p] - log p(o)
  - Verificar se F diminui ao longo do tempo (sistema aprende)
  - **Critério de Sucesso:** F decresce monotonicamente após 20+ timesteps
- [ ] **3.1.10** [P3] Implementar "Active Inference" — o modelo age para minimizar surpresa:
  - O modelo não apenas prevê, mas escolhe ação que minimiza F futuro
  - **Critério de Sucesso:** Modelo com Active Inference tem F menor que modelo passivo

#### 3.2 — Multiverso Computacional (lab07-delta-branches)

- [ ] **3.2.0** [P1] Criar pasta `pesquisa0/labs/lab07-delta-branches/`
- [ ] **3.2.1** [P1] Implementar "Delta Branch Store" em Python:
  - Base state: array numpy de 1MB
  - Branch: apenas os índices e valores que diferem da base
  - Operações: criar branch, ler valor (fallback para base), merge, delete
  - **Critério de Sucesso:** CRUD funcional + testes unitários passando
- [ ] **3.2.2** [P1] Benchmark de memória: Delta vs Cópia Completa
  - Criar 100 branches com 0.01%, 0.1%, 1%, 10% de diferença
  - **Critério de Sucesso:** Delta usa <10% da memória da cópia completa para divergência <1%
- [ ] **3.2.3** [P2] Implementar "Colapso" — quando dado real chega, descartar branches incompatíveis
  - **Critério de Sucesso:** Tempo de colapso <1ms para 100 branches
- [ ] **3.2.4** [P2] Implementar usando XOR Delta do Crompressor (Go) em vez de numpy
  - **Critério de Sucesso:** Performance igual ou melhor que numpy + integração com motor .crom
- [ ] **3.2.5** [P3] Benchmark: quantas branches simultâneas cabem em 1GB, 4GB, 8GB de RAM?
  - **Critério de Sucesso:** Tabela de capacidade por tamanho de estado base
- [ ] **3.2.6** [P3] Implementar Merkle Tree parcial para verificação rápida de integridade de branches
  - **Critério de Sucesso:** Verificação de integridade em O(log N) onde N = chunks alterados

---

### ═══════════════════════════════════════════════════════════
### EIXO 04 — FÍSICA MULTIDIMENSIONAL (VERIFICAÇÃO)
### ═══════════════════════════════════════════════════════════

#### 4.1 — Dimensionalidade Intrínseca dos Embeddings (lab04-dimensionalidade-embeddings)

**Objetivo:** Medir quantas "dimensões efetivas" o Codebook Learning realmente usa, validando a analogia com compactificação dimensional.

##### Configuração

- [ ] **4.1.0** [P1] Criar pasta `pesquisa0/labs/lab04-dimensionalidade-embeddings/`
- [ ] **4.1.1** [P1] Criar `medir_dimensionalidade.py` — carregar codebooks treinados do tensor-vivo (exp2) e medir dimensionalidade intrínseca:
  - PCA: quantos componentes capturam 95% da variância?
  - MDS (Multidimensional Scaling): qual é a dimensão mínima para representar distâncias?
  - Estimador de Two-NN (Facco et al., 2017): dimensionalidade intrínseca sem PCA
  - **Critério de Sucesso:** Número concreto de "dimensões efetivas" para K=128, K=256, K=512
- [ ] **4.1.2** [P1] Comparar dimensionalidade de codebooks de diferentes arquiteturas:
  - Codebook MNIST MLP (exp2) vs CIFAR-10 CNN (exp3) vs GPT-2 (exp5)
  - **Critério de Sucesso:** Tabela mostrando se arquiteturas maiores usam mais dimensões efetivas
- [ ] **4.1.3** [P2] Visualizar codebooks em 2D/3D via t-SNE ou UMAP:
  - Verificar se existem clusters naturais (análogo a "variedades compactadas")
  - **Critério de Sucesso:** Plot salvo como PNG mostrando estrutura (ou ausência de estrutura)
- [ ] **4.1.4** [P2] Calcular "taxa de compactação dimensional":
  - Dim original dos pesos (ex: 768 para GPT-2) vs dim efetiva do codebook
  - Ratio = dim_original / dim_efetiva — análogo à compactificação Calabi-Yau
  - **Critério de Sucesso:** Número calculado e comparado com ratio 10D→4D da física (2.5x)

##### Validação de Analogias Físicas

- [ ] **4.1.5** [P2] Testar "analogia Calabi-Yau": a informação está "enrolada" em poucas dimensões?
  - Treinar codebook com diferentes K e medir dimensionalidade intrínseca
  - Se dim_efetiva ≈ constante independente de K → suporta analogia
  - **Critério de Sucesso:** Gráfico K vs dim_efetiva com conclusão
- [ ] **4.1.6** [P3] Testar "analogia de sombras" (Teoria-S 13D):
  - Projetar codebook em diferentes subespaços e verificar se projeções diferentes parecem dados diferentes mas vêm da mesma fonte
  - **Critério de Sucesso:** Correlação entre projeções documentada
- [ ] **4.1.7** [P3] Medir "simetrias internas" do codebook (análogo a Heteróticas):
  - Existem transformações que preservam a accuracy? (rotação, reflexão, permutação)
  - **Critério de Sucesso:** Lista de invariâncias encontradas (ou ausência delas)
- [ ] **4.1.8** [P3] Calcular entropia de Shannon dos índices do codebook vs entropia dos pesos originais:
  - Se codebook reduz entropia significativamente → compactação dimensional confirmada
  - **Critério de Sucesso:** Ratio de entropia documentado

#### 4.2 — Escada WLM no Tensor-Vivo (Verificação Teórica)

**Objetivo:** Mapear as 5 fases da escada dimensional WLM (8D-12D) no processo de Codebook Learning.

- [ ] **4.2.1** [P2] Documentar se Codebook Learning possui 8D (recursão): output do codebook vira input do próximo epoch?
  - **Critério de Sucesso:** Diagrama do fluxo de dados mostrando feedback loop
- [ ] **4.2.2** [P2] Documentar se possui 9D (transparência): existe logging/observabilidade da convergência?
  - **Critério de Sucesso:** Verificar se loss curves e métricas estão sendo registradas
- [ ] **4.2.3** [P2] Documentar se possui 10D (estabilidade): existem mecanismos anti-divergência?
  - Learning rate schedulers, gradient clipping, early stopping
  - **Critério de Sucesso:** Lista de mecanismos de estabilidade no código existente
- [ ] **4.2.4** [P2] Documentar se possui 11D (multicamadas): camadas se comunicam durante training?
  - **Critério de Sucesso:** Análise do grafo computacional do CodebookLinear
- [ ] **4.2.5** [P2] Documentar se possui 12D (fechamento): o .crom final é autocontido e selado?
  - **Critério de Sucesso:** Verificar se o formato brain.crom é self-describing

---

### ═══════════════════════════════════════════════════════════
### EIXO 05 — IA DIMENSIONAL: LLM 4D → IA 5D
### ═══════════════════════════════════════════════════════════

#### 5.1 — Tree of Thoughts vs Autoregressivo (lab05-tot-vs-autoregressive)

**Objetivo:** Implementar ToT mínimo e comparar com geração autorregressiva pura em tarefas de raciocínio.

##### Configuração

- [ ] **5.1.0** [P1] Criar pasta `pesquisa0/labs/lab05-tot-vs-autoregressive/`
- [ ] **5.1.1** [P1] Criar `tot_miniatura.py` — implementação mínima de Tree of Thoughts:
  - Tarefa: resolver equação aritmética complexa (ex: 24 game)
  - Modelo base: MLP treinado ou API de LLM local (ollama)
  - Gerar 3 branches de pensamento por nó
  - Avaliar cada branch com heurística simples
  - **Critério de Sucesso:** ToT resolve >70% dos problemas vs <30% para geração linear
- [ ] **5.1.2** [P1] Implementar avaliação por "Sistema 2":
  - O modelo julga seus próprios pensamentos antes de prosseguir
  - **Critério de Sucesso:** Accuracy com auto-avaliação > accuracy sem auto-avaliação
- [ ] **5.1.3** [P1] Medir overhead computacional do ToT vs geração linear:
  - Tempo de execução, memória, tokens gerados
  - **Critério de Sucesso:** Tabela comparativa com ratio custo/benefício

##### Integração com Crompressor

- [ ] **5.1.4** [P2] Implementar ToT com Delta Storage para branches:
  - Cada branch armazena apenas o Delta vs branch pai
  - **Critério de Sucesso:** >50% redução de memória vs cópia completa dos estados
- [ ] **5.1.5** [P2] Implementar pruning com threshold de D_KL:
  - Descartar branches cuja divergência excede threshold
  - **Critério de Sucesso:** Pruning reduz branches ativas em >60% sem perder a branch ótima
- [ ] **5.1.6** [P3] Comparar formalmente: ToT + Delta vs ToT puro vs Autogressivo puro
  - Métricas: accuracy, latência, memória, tokens gerados
  - **Critério de Sucesso:** Relatório JSON com as 3 configurações

#### 5.2 — Codebook Learning no KV Cache (lab06-kv-cache-codebook)

**Objetivo:** Aplicar Codebook Learning (validado no tensor-vivo) diretamente ao KV Cache de um Transformer para comprimir contexto longo.

##### Configuração

- [ ] **5.2.0** [P1] Criar pasta `pesquisa0/labs/lab06-kv-cache-codebook/`
- [ ] **5.2.1** [P1] Criar `kv_cache_analysis.py` — extrair e analisar KV Cache de um Transformer pequeno:
  - Modelo: GPT-2 small (ou distilgpt2) via HuggingFace
  - Extrair Key e Value tensors de todas as camadas
  - Medir: dimensionalidade, entropia, redundância entre camadas
  - **Critério de Sucesso:** Relatório JSON com estatísticas do KV Cache
- [ ] **5.2.2** [P1] Aplicar K-Means quantization (como exp1 do tensor-vivo) ao KV Cache:
  - Testar K=64, 128, 256, 512 com blocos de tamanho 8, 16, 32
  - Medir: perplexity antes vs depois da quantização
  - **Critério de Sucesso:** <5% aumento de perplexity com K=256
- [ ] **5.2.3** [P2] Implementar CodebookLinear para KV Cache (reusar código do tensor-vivo):
  - Substituir KV Cache dinâmico por Codebook + índices
  - Medir: redução de VRAM, impacto na perplexity
  - **Critério de Sucesso:** >10x redução de VRAM com <10% perda de perplexity
- [ ] **5.2.4** [P2] Benchmark de contexto longo:
  - Baseline: GPT-2 com contexto de 256, 512, 1024 tokens
  - Codebook: mesmo modelo com KV Cache comprimido
  - **Critério de Sucesso:** Modelo com codebook suporta contexto 4x maior na mesma VRAM
- [ ] **5.2.5** [P2] Comparar com técnicas existentes de compressão de KV Cache:
  - Pesquisar: Activation Beacon, GQA, MQA
  - **Critério de Sucesso:** Tabela comparativa (compressão, perplexity, velocidade)
- [ ] **5.2.6** [P3] Testar se codebook treinado num contexto funciona em outro contexto (transferibilidade):
  - Treinar codebook em texto A, aplicar em texto B
  - **Critério de Sucesso:** Perplexity em B com codebook de A vs codebook específico de B

---

### ═══════════════════════════════════════════════════════════
### EIXO 06 — INTEGRAÇÃO COM ECOSSISTEMA CROMPRESSOR
### ═══════════════════════════════════════════════════════════

#### 6.1 — Firewall de Realidade (lab08-sandbox-alucinacao)

**Objetivo:** Implementar protótipo de sandbox que detecta e bloqueia "alucinações" usando métricas do Crompressor.

- [ ] **6.1.0** [P1] Criar pasta `pesquisa0/labs/lab08-sandbox-alucinacao/`
- [ ] **6.1.1** [P1] Criar `detector_alucinacao.py`:
  - Input: sequência de tokens gerados por LLM
  - Métrica: Delta Ratio entre cada token e o Codebook de domínio
  - Se Delta Ratio > threshold → flag como "alucinação potencial"
  - **Critério de Sucesso:** Detecta >80% de fatos inventados em texto gerado
- [ ] **6.1.2** [P1] Criar dataset de teste:
  - 50 afirmações verdadeiras + 50 afirmações alucinadas
  - **Critério de Sucesso:** Dataset JSON criado e validado manualmente
- [ ] **6.1.3** [P2] Implementar sandbox de isolamento:
  - Branches de simulação rodam em "espaço isolado"
  - Resultados só passam para memória principal se D_KL < threshold
  - **Critério de Sucesso:** Zero contaminação de branches descartadas na memória principal
- [ ] **6.1.4** [P2] Calibrar threshold de D_KL:
  - Muito baixo = criatividade bloqueada (falsos positivos)
  - Muito alto = alucinações passam (falsos negativos)
  - **Critério de Sucesso:** Curva ROC com ponto ótimo identificado
- [ ] **6.1.5** [P3] Integrar com crompressor-security (Ed25519 sign):
  - Assinar estados verificados para impedir alteração retroativa
  - **Critério de Sucesso:** Estado assinado → tentativa de alteração → rejeição

#### 6.2 — Protocolo Sinapse (lab09-sinapse-protocolo)

**Objetivo:** Implementar protocolo de comunicação entre branches simuladas (analogia com 11D multicamadas).

- [ ] **6.2.0** [P1] Criar pasta `pesquisa0/labs/lab09-sinapse-protocolo/`
- [ ] **6.2.1** [P1] Definir protocolo de mensagens:
  - `DELTA_UPDATE`: branch atualizou estado
  - `DIVERGENCE_ALERT`: D_KL acima do threshold
  - `COLLAPSE_SIGNAL`: dado real chegou
  - `MERGE_REQUEST`: branches convergiram
  - **Critério de Sucesso:** Spec JSON/protobuf documentada
- [ ] **6.2.2** [P1] Implementar em Python (asyncio) com 5 branches simuladas:
  - Cada branch roda em coroutine separada
  - Orquestrador central recebe mensagens e gerencia lifecycle
  - **Critério de Sucesso:** 5 branches comunicando, collapse funcional
- [ ] **6.2.3** [P2] Implementar em Go (goroutines + channels):
  - Mesma lógica, mas com performance nativa
  - **Critério de Sucesso:** Benchmark Go vs Python (esperado: Go 10x+ mais rápido)
- [ ] **6.2.4** [P2] Testar escalabilidade: 10, 50, 100, 500 branches simultâneas
  - **Critério de Sucesso:** Gráfico de latência e memória vs número de branches
- [ ] **6.2.5** [P3] Integrar com protocolo P2P do Crompressor existente
  - **Critério de Sucesso:** Branches distribuídas entre 2+ processos comunicando via P2P

#### 6.3 — Loop de Active Inference (lab10-active-inference-loop)

**Objetivo:** Implementar o loop completo de Inferência Ativa do Friston como agente CROM.

- [ ] **6.3.0** [P1] Criar pasta `pesquisa0/labs/lab10-active-inference-loop/`
- [ ] **6.3.1** [P1] Criar `active_inference_agent.py`:
  - Ambiente: grid 2D com objetivo (chegar ao ponto X)
  - Agente tem: World Model interno + sensores (posição real)
  - Loop: Prever → Observar → Calcular F → Agir para minimizar F
  - **Critério de Sucesso:** Agente navega até o objetivo usando Active Inference
- [ ] **6.3.2** [P1] Comparar com agente baseline (random walk):
  - **Critério de Sucesso:** AI agent chega ao objetivo 5x mais rápido que random
- [ ] **6.3.3** [P2] Adicionar "surpresas" ao ambiente (obstáculos que aparecem):
  - Medir tempo de adaptação do World Model
  - **Critério de Sucesso:** F sobe brevemente, depois volta a cair (adaptação)
- [ ] **6.3.4** [P2] Integrar Codebook como memória do World Model:
  - Em vez de armazenar o mapa inteiro, armazena Codebook + Deltas
  - **Critério de Sucesso:** Agente funciona igualmente com 90% menos memória
- [ ] **6.3.5** [P3] Implementar com múltiplas branches (MCTS + Active Inference):
  - O agente simula 10 ações futuras antes de agir
  - **Critério de Sucesso:** Accuracy de navegação aumenta >20% vs sem branches

#### 6.4 — Dual Clock (lab12-dual-clock)

**Objetivo:** Implementar sistema com dois vetores temporais (análogo à Teoria-F 12D).

- [ ] **6.4.0** [P2] Criar pasta `pesquisa0/labs/lab12-dual-clock/`
- [ ] **6.4.1** [P2] Implementar `dual_clock.py`:
  - Clock 1 (Inercial): avança com o tempo real do processador
  - Clock 2 (Prospectivo): avança explorando futuros possíveis
  - Sincronização: quando dados reais chegam, Clock 2 reseta para Clock 1
  - **Critério de Sucesso:** Dois clocks rodando, sincronização funcional
- [ ] **6.4.2** [P2] Medir "vantagem temporal": quantos timesteps à frente o Clock 2 consegue explorar
  - **Critério de Sucesso:** Clock 2 explora >100 timesteps enquanto Clock 1 avança 1
- [ ] **6.4.3** [P3] Integrar dual clock com World Model do lab03:
  - Clock 1 roda o modelo real, Clock 2 roda simulações de branches
  - **Critério de Sucesso:** Sistema dual-clock tem melhor accuracy que single-clock

---

### ═══════════════════════════════════════════════════════════
### VALIDAÇÃO CRUZADA ENTRE EIXOS
### ═══════════════════════════════════════════════════════════

#### 7.1 — Cruzamento Eixo 01 × Eixo 03 (FPS × World Model)

- [ ] **7.1.1** [P2] Verificar se FPS mais alto (lab01) permite World Model mais preciso (lab03)
  - Rodar World Model com 10 Hz vs 1000 Hz de atualização
  - **Critério de Sucesso:** Modelo 1000Hz tem erro <50% do modelo 10Hz
- [ ] **7.1.2** [P3] Calcular "banda mínima" para World Model funcional
  - Qual é o FPS mínimo para manter erro <5%?
  - **Critério de Sucesso:** Número de Hz mínimo documentado

#### 7.2 — Cruzamento Eixo 02 × Eixo 06 (Observadores × Sinapse)

- [ ] **7.2.1** [P2] Usar protocolo sinapse (lab09) para comunicação entre observadores (lab02)
  - Observadores enviam DELTA_UPDATE em vez de dados brutos
  - **Critério de Sucesso:** Merge funciona com protocolo sinapse + >80% redução bandwidth
- [ ] **7.2.2** [P3] Testar colapso de observadores (COLLAPSE_SIGNAL) quando dado real contradiz merge
  - **Critério de Sucesso:** Sistema rejeita observadores inconsistentes em <10ms

#### 7.3 — Cruzamento Eixo 03 × Eixo 05 (World Model × ToT)

- [ ] **7.3.1** [P2] Usar World Model (lab03) como avaliador de branches no ToT (lab05)
  - Em vez de heurística, World Model prevê consequência de cada branch
  - **Critério de Sucesso:** ToT+WorldModel > ToT+heurística em accuracy
- [ ] **7.3.2** [P3] Comprimir World Model com Codebook (lab06) e verificar que ToT continua funcionando
  - **Critério de Sucesso:** ToT com World Model comprimido tem <5% perda vs não-comprimido

#### 7.4 — Cruzamento Eixo 04 × Eixo 06 (Dimensões × Codebook)

- [ ] **7.4.1** [P2] Correlacionar dimensionalidade intrínseca (lab04) com compressão do KV Cache (lab06)
  - Se dim_efetiva é baixa, compressão deveria ser alta
  - **Critério de Sucesso:** Correlação negativa entre dim_efetiva e ratio de compressão
- [ ] **7.4.2** [P3] Verificar se a escada WLM (8D-12D) se aplica ao ciclo de vida de um agente CROM completo
  - **Critério de Sucesso:** Documento mapeando cada dimensão para fase do agente

#### 7.5 — Cruzamento Eixo 05 × Eixo 06 (IA 5D × Active Inference)

- [ ] **7.5.1** [P2] Integrar ToT (lab05) com Active Inference (lab10):
  - ToT gera branches, Active Inference decide qual seguir minimizando F
  - **Critério de Sucesso:** Sistema integrado supera ambos isolados
- [ ] **7.5.2** [P3] Agente CROM completo com todos os componentes:
  - neurônio (memória) + sinapse (comunicação) + security (filtro) + ia (decisão)
  - **Critério de Sucesso:** Agente navega ambiente complexo usando todos os labs integrados

---

### ═══════════════════════════════════════════════════════════
### DOCUMENTAÇÃO E RELATÓRIO FINAL
### ═══════════════════════════════════════════════════════════

#### 8.1 — Documentação de Resultados

- [ ] **8.1.1** [P1] Criar `pesquisa0/resultados/` com JSONs de output de cada lab
- [ ] **8.1.2** [P1] Cada lab gera arquivo `resultados/labXX_results.json` com:
  - Timestamp, hardware usado, parâmetros, métricas, conclusão
  - **Critério de Sucesso:** JSON schema consistente entre todos os labs
- [ ] **8.1.3** [P2] Gerar tabela resumo cruzando todos os labs:
  - Hipótese → Lab → Resultado → Veredicto
  - **Critério de Sucesso:** Tabela similar à CONCLUSOES.md do tensor-vivo
- [ ] **8.1.4** [P2] Criar `pesquisa0/CONCLUSOES.md` com veredicto final:
  - Quais analogias dimensionais foram validadas?
  - Quais foram refutadas?
  - Quais precisam de mais dados?
  - **Critério de Sucesso:** Documento com tabela H1-HN e veredictos

#### 8.2 — Atualização do Repositório

- [ ] **8.2.1** [P2] Atualizar `README.md` raiz do projeto com link para pesquisa0
- [ ] **8.2.2** [P2] Atualizar `docs/09-ROADMAP.md` com itens da pesquisa0
- [ ] **8.2.3** [P3] Criar referências cruzadas entre pesquisa0 e tensor-vivo
- [ ] **8.2.4** [P3] Publicar resultados significativos como Issues no GitHub

#### 8.3 — Relatório Final de Sinergia

- [ ] **8.3.1** [P1] Análise de viabilidade: quais labs são viáveis no hardware local (ThinkPad X230)?
  - **Critério de Sucesso:** Lista de labs viáveis vs que requerem cloud
- [ ] **8.3.2** [P2] Principais riscos:
  - Risco 1: Analogias dimensionais podem ser apenas metáforas sem substância computacional
  - Risco 2: Codebook Learning no KV Cache pode não escalar para modelos >1B params
  - Risco 3: Active Inference pode ser computacionalmente caro demais para Edge
  - **Critério de Sucesso:** Mitigação documentada para cada risco
- [ ] **8.3.3** [P2] Próximos passos recomendados após conclusão dos labs:
  - Se validados: integrar ao motor .crom como funcionalidade nativa
  - Se refutados: documentar por que e propor direções alternativas
  - **Critério de Sucesso:** Roadmap de 3 meses pós-labs

---

## 📊 RESUMO QUANTITATIVO DO CHECKLIST

| Eixo | Items P1 | Items P2 | Items P3 | Total |
|:-----|:---------|:---------|:---------|:------|
| 01 — Percepção Temporal | 4 | 6 | 6 | **16** |
| 02 — Observadores | 4 | 6 | 4 | **14** |
| 03 — Simulação/World Models | 9 | 7 | 4 | **20** |
| 04 — Dimensões | 3 | 8 | 4 | **15** |
| 05 — IA Dimensional | 5 | 6 | 3 | **14** |
| 06 — Integração Crompressor | 7 | 9 | 6 | **22** |
| 07 — Validação Cruzada | 0 | 6 | 5 | **11** |
| 08 — Documentação | 3 | 6 | 3 | **12** |
| **TOTAL** | **35** | **54** | **35** | **124** |

### Ordem de Execução Recomendada

```
FASE 1 (P1 — Fundação):    35 items
├── lab01 → lab03 → lab07  (benchmarks + world model + branches)
├── lab05 → lab06          (ToT + KV cache)
└── lab08 → lab09 → lab10  (sandbox + sinapse + active inference)

FASE 2 (P2 — Expansão):    54 items
├── Análises de dimensionalidade (lab04)
├── Validações cruzadas (eixo 7)
└── Escalonamento e benchmarks avançados

FASE 3 (P3 — Fronteira):   35 items
├── Analogias físicas avançadas
├── Integrações completas
└── Relatório final e publicação
```

---

> **Tamanho total deste checklist:** 124 items em 12 labs, cobrindo 6 eixos + validação cruzada + documentação.
> **Cada item tem:** ID único, prioridade, descrição detalhada e critério de sucesso quantificável.
> **Estimativa de esforço:** ~2-4 semanas para Fase 1, ~2-3 semanas para Fase 2, ~2 semanas para Fase 3.

---

*"O neurônio que comprime é o neurônio que pensa."*
