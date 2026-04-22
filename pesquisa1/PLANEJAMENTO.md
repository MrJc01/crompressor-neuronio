# PLANEJAMENTO — Pesquisa1: Do Conceito ao Motor

> **Objetivo:** Transformar os resultados validados da pesquisa0 em um **motor CROM 5D funcional em Go**, validado em escala com modelos >1B params, pronto para produção.
>
> **Data:** 2026-04-22
> **Status:** ⏳ NÃO INICIADA — 0/150 items
> **Pré-requisito:** pesquisa0 encerrada (129/129, 15/16 hipóteses)

---

## Estrutura do Projeto

```
pesquisa1/
├── README.md
├── PLANEJAMENTO.md                      ← Este arquivo
├── CONCLUSOES.md
├── REFERENCIAS.md                       Levantamento de 18 papers (2025-2026)
├── 01-motor-go-nativo/                  Teoria: motor Go, interfaces, performance
├── 02-escala-validacao/                 Teoria: LLaMA-7B, CommVQ, benchmarks
├── 03-seguranca-deteccao/              Teoria: ensemble detector, streaming
├── 04-distribuicao-p2p/                Teoria: delta P2P, multi-agente
├── 05-deployment/                       Teoria: WASM, ARM, paper arXiv
├── diagramas/                           3 diagramas mermaid
├── papers/                              Papers gerados (papel0..N)
├── labs/
│   ├── lab13-agent-crom-v2/            Motor Go completo
│   ├── lab14-kv-cache-llama/           LLaMA-7B real (Colab)
│   ├── lab15-ensemble-detector/        P≥90% R≥95%
│   ├── lab16-codebook-rope/            CommVQ + PCA
│   ├── lab17-codebook-training-go/     VQ training em Go
│   ├── lab18-delta-p2p/                P2P protocol
│   ├── lab19-crom-format-v2/           Formato binário .crom v2
│   ├── lab20-streaming-inference/      Inference + detecção real-time
│   ├── lab21-multi-agent-crom/         N agentes cooperativos
│   ├── lab22-benchmark-suite/          Benchmarks vs SOTA
│   ├── lab23-edge-deployment/          WASM + ARM
│   └── lab24-paper-arxiv/              Draft publicação
└── resultados/                          JSONs de cada lab
```

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
## EIXO 01 — MOTOR GO NATIVO (<1ms/step)
## ═══════════════════════════════════════════════════════════

### 1.1 — Agente CROM v2 (lab13-agent-crom-v2)

**Objetivo:** Portar o pipeline completo Python (blitz_final.py) para Go puro com <0.5ms/step.

- [ ] **1.1.0** [P1] Criar pasta `pesquisa1/labs/lab13-agent-crom-v2/`
- [ ] **1.1.1** [P1] Definir interfaces Go: Agent, WorldModel, BranchManager, Firewall
  - **Critério de Sucesso:** Interfaces compilam, testes de interface passam
- [ ] **1.1.2** [P1] Implementar WorldModel com EMA (Exponential Moving Average)
  - Input: observation []float64 → Output: prediction []float64
  - **Critério de Sucesso:** Erro <5% em sequência sintética (igual pesquisa0)
- [ ] **1.1.3** [P1] Implementar BranchManager usando DeltaBranchStore existente
  - Explore(depth=3, branches=5) → gera 15 futuros possíveis
  - Collapse: seleciona branch com menor free energy
  - **Critério de Sucesso:** 15 branches em <1ms
- [ ] **1.1.4** [P1] Implementar Decision com weighted collapse por variância
  - **Critério de Sucesso:** Agente navega ambiente 1D (igual pesquisa0 Lab10)
- [ ] **1.1.5** [P1] Implementar Firewall com threshold de erro
  - Bloqueia predições com erro > limiar
  - **Critério de Sucesso:** ≥70% de alucinações bloqueadas (igual pesquisa0)
- [ ] **1.1.6** [P1] Integrar Ed25519 para assinar cada decisão
  - **Critério de Sucesso:** Sign <50μs, Verify <200μs
- [ ] **1.1.7** [P2] Pipeline completo: Sensor→WM→Branches→Decision→Firewall
  - **Critério de Sucesso:** <0.5ms/step, 200 steps sem crash
- [ ] **1.1.8** [P2] Benchmark: Go v2 vs Python pesquisa0
  - Medir: latência, memória, throughput
  - **Critério de Sucesso:** ≥10x speedup sobre Python
- [ ] **1.1.9** [P2] Adicionar Kalman Filter como alternativa ao EMA
  - **Critério de Sucesso:** Erro <3% (melhor que EMA)
- [ ] **1.1.10** [P3] Profiling com pprof: identificar hotspots
  - **Critério de Sucesso:** Relatório de CPU/memória, zero allocations no hot path
- [ ] **1.1.11** [P3] Implementar pool de goroutines para branches paralelas
  - **Critério de Sucesso:** Exploração paralela, <0.3ms para 15 branches

### 1.2 — Codebook Training em Go (lab17-codebook-training-go)

**Objetivo:** Implementar K-Means e Vector Quantization em Go puro (sem CGo, sem Python).

- [ ] **1.2.0** [P1] Criar pasta `pesquisa1/labs/lab17-codebook-training-go/`
- [ ] **1.2.1** [P1] Implementar K-Means clustering em Go
  - Input: [][]float64 → Output: centroids [][]float64, assignments []int
  - **Critério de Sucesso:** Converge em dados sintéticos, resultados iguais ao sklearn
- [ ] **1.2.2** [P1] Implementar VQ encoder: dado vector, retornar índice do centróide mais próximo
  - **Critério de Sucesso:** O(K×D) por lookup, <10μs para K=256, D=64
- [ ] **1.2.3** [P2] Implementar batch training: treinar codebook em streaming
  - **Critério de Sucesso:** Codebook estável após 10K vectors
- [ ] **1.2.4** [P2] Serializar codebook para formato binário (.crom)
  - **Critério de Sucesso:** Load <1ms para K=256, D=64
- [ ] **1.2.5** [P3] Mini-batch K-Means para datasets grandes
  - **Critério de Sucesso:** 1M vectors em <10s

### 1.3 — Formato .crom v2 (lab19-crom-format-v2)

**Objetivo:** Definir formato binário para codebooks neurais, compatível com o .crom existente.

- [ ] **1.3.0** [P1] Criar pasta `pesquisa1/labs/lab19-crom-format-v2/`
- [ ] **1.3.1** [P1] Definir header binário: magic bytes, versão, K, D, metadata
  - **Critério de Sucesso:** Spec documentada, parser Go funcional
- [ ] **1.3.2** [P2] Serialização/deserialização de codebook + deltas
  - **Critério de Sucesso:** Round-trip sem perda, <1ms I/O
- [ ] **1.3.3** [P2] Compatibilidade com neuronio.go existente (CromHeader)
  - **Critério de Sucesso:** Integração com BrainCrom struct
- [ ] **1.3.4** [P3] Compressão adicional do codebook com zstd
  - **Critério de Sucesso:** 30%+ redução adicional

---

## ═══════════════════════════════════════════════════════════
## EIXO 02 — ESCALA E VALIDAÇÃO (>1B PARAMS)
## ═══════════════════════════════════════════════════════════

### 2.1 — KV Cache com LLaMA-7B (lab14-kv-cache-llama)

**Objetivo:** Validar compressão do KV Cache em modelo >1B params com perplexity real.

- [ ] **2.1.0** [P1] Criar pasta `pesquisa1/labs/lab14-kv-cache-llama/`
- [ ] **2.1.1** [P1] Script Colab: carregar LLaMA-7B (4-bit) no T4
  - **Critério de Sucesso:** Modelo carrega sem OOM no Colab free
- [ ] **2.1.2** [P1] Extrair KV Cache real do LLaMA-7B
  - **Critério de Sucesso:** Tensores shape (batch, heads, seq, dim) salvos
- [ ] **2.1.3** [P1] Aplicar codebook VQ (K=256) ao KV Cache real
  - **Critério de Sucesso:** Compressão ≥90%, cosine similarity ≥0.85
- [ ] **2.1.4** [P1] Medir perplexity antes/depois da compressão
  - **Critério de Sucesso:** Perplexity increase <2%
- [ ] **2.1.5** [P2] Testar K=64, 128, 256, 512 — curva de Pareto
  - **Critério de Sucesso:** Gráfico compressão vs perplexity
- [ ] **2.1.6** [P2] Comparar com TurboQuant (3-bit, data-oblivious)
  - **Critério de Sucesso:** Tabela comparativa documentada
- [ ] **2.1.7** [P3] Testar com contexto longo (4K, 8K tokens)
  - **Critério de Sucesso:** Compressão mantém >80% em seq=8K

### 2.2 — Codebook RoPE-Comutativo (lab16-codebook-rope)

**Objetivo:** Implementar CommVQ-style codebook que comuta com Rotary Position Embedding.

- [ ] **2.2.0** [P1] Criar pasta `pesquisa1/labs/lab16-codebook-rope/`
- [ ] **2.2.1** [P1] Estudar paper CommVQ (Apple) — entender comutatividade
  - **Critério de Sucesso:** Documento explicando RoPE + VQ
- [ ] **2.2.2** [P1] Implementar RoPE em Python/NumPy
  - **Critério de Sucesso:** Output igual ao transformers.models.llama
- [ ] **2.2.3** [P2] Implementar codebook comutativo: VQ(RoPE(x)) = RoPE(VQ(x))
  - **Critério de Sucesso:** Erro de comutatividade <1%
- [ ] **2.2.4** [P2] Adicionar PCA decorrelation (KVTC-style) antes do VQ
  - **Critério de Sucesso:** Compressão adicional ≥2x sobre VQ puro
- [ ] **2.2.5** [P2] Pipeline completo: PCA → CommVQ → Delta → Entropy
  - **Critério de Sucesso:** >50x compressão em dados GPT-2
- [ ] **2.2.6** [P3] Validar com LLaMA-7B no Colab
  - **Critério de Sucesso:** >80x compressão com perplexity <3% increase

### 2.3 — Benchmark Suite (lab22-benchmark-suite)

**Objetivo:** Criar suite de benchmarks comparando Crompressor vs SOTA.

- [ ] **2.3.0** [P2] Criar pasta `pesquisa1/labs/lab22-benchmark-suite/`
- [ ] **2.3.1** [P2] Implementar benchmarks: latência, memória, throughput
  - **Critério de Sucesso:** Script reproduzível com output JSON
- [ ] **2.3.2** [P2] Comparar vs TurboQuant, KVTC, CommVQ (tabela)
  - **Critério de Sucesso:** Tabela publicável
- [ ] **2.3.3** [P2] Medir dim intrínseca com lFCI (Fadanni 2026)
  - **Critério de Sucesso:** lFCI vs MLE: concordância ±20%
- [ ] **2.3.4** [P3] Codebook K-adaptativo baseado em dim intrínseca medida
  - **Critério de Sucesso:** K ajustado automaticamente, compressão ≥10% melhor
- [ ] **2.3.5** [P3] Benchmark em LLM-KICK (Apple 2025) para validação completa
  - **Critério de Sucesso:** Scores publicáveis

---

## ═══════════════════════════════════════════════════════════
## EIXO 03 — SEGURANÇA E DETECÇÃO
## ═══════════════════════════════════════════════════════════

### 3.1 — Ensemble Detector (lab15-ensemble-detector)

**Objetivo:** Combinar v1 (n-grams) + v3 (SBERT) para maximizar P e R simultaneamente.

- [ ] **3.1.0** [P1] Criar pasta `pesquisa1/labs/lab15-ensemble-detector/`
- [ ] **3.1.1** [P1] Implementar voting scheme: v3 filtra (R=100%) + v1 contra-prova (P=100%)
  - **Critério de Sucesso:** P≥90%, R≥95%
- [ ] **3.1.2** [P2] Adicionar TF-IDF (v2) como terceiro votante
  - **Critério de Sucesso:** F1 ≥ 90%
- [ ] **3.1.3** [P2] Benchmark contra SelfCheckGPT e SAFE
  - **Critério de Sucesso:** Tabela comparativa em 50+ test cases
- [ ] **3.1.4** [P2] Detector com dataset multilíngue (PT-BR + EN)
  - **Critério de Sucesso:** Performance ≥85% em PT-BR
- [ ] **3.1.5** [P3] Calibração de threshold por domínio
  - **Critério de Sucesso:** Thresholds ótimos para 3+ domínios

### 3.2 — Streaming Inference (lab20-streaming-inference)

**Objetivo:** Detectar alucinações em tempo real durante a geração de tokens.

- [ ] **3.2.0** [P2] Criar pasta `pesquisa1/labs/lab20-streaming-inference/`
- [ ] **3.2.1** [P2] Implementar token-by-token detection pipeline
  - A cada token gerado, calcular distância ao codebook
  - **Critério de Sucesso:** Detecção em <1ms por token
- [ ] **3.2.2** [P2] Integrar com KV Cache comprimido: detecção usa os mesmos índices
  - **Critério de Sucesso:** Zero overhead adicional de memória
- [ ] **3.2.3** [P3] Auto-correção: ao detectar alucinação, regenerar a partir do último token seguro
  - **Critério de Sucesso:** Demonstração funcional em 10 exemplos
- [ ] **3.2.4** [P3] Implementar em Go nativo
  - **Critério de Sucesso:** <0.5ms por token, 100% recall mantido

---

## ═══════════════════════════════════════════════════════════
## EIXO 04 — DISTRIBUIÇÃO P2P E MULTI-AGENTE
## ═══════════════════════════════════════════════════════════

### 4.1 — Delta P2P (lab18-delta-p2p)

**Objetivo:** Distribuir deltas entre processos/máquinas via protocolo P2P.

- [ ] **4.1.0** [P2] Criar pasta `pesquisa1/labs/lab18-delta-p2p/`
- [ ] **4.1.1** [P2] Implementar protocolo P2P básico em Go (TCP/UDP)
  - **Critério de Sucesso:** 2 processos trocam deltas
- [ ] **4.1.2** [P2] Adicionar 1-bit delta scheme para reduzir bandwidth
  - **Critério de Sucesso:** ≥90% redução de bandwidth vs delta raw
- [ ] **4.1.3** [P2] Implementar discovery: peers se encontram em LAN
  - **Critério de Sucesso:** Auto-discovery funcional em rede local
- [ ] **4.1.4** [P3] Integrar Ed25519 para autenticação de deltas
  - **Critério de Sucesso:** Delta rejeitado se assinatura inválida
- [ ] **4.1.5** [P3] Benchmark: latência e throughput para 3, 5, 10 peers
  - **Critério de Sucesso:** Gráfico de escalabilidade

### 4.2 — Multi-Agente CROM (lab21-multi-agent-crom)

**Objetivo:** Múltiplos agentes CROM colaborando para decisões coletivas.

- [ ] **4.2.0** [P2] Criar pasta `pesquisa1/labs/lab21-multi-agent-crom/`
- [ ] **4.2.1** [P2] Implementar FREE_ENERGY_SHARE no protocolo Sinapse
  - Cada agente broadcast sua free energy F
  - **Critério de Sucesso:** 3 agentes sincronizam F
- [ ] **4.2.2** [P2] Consenso epistêmico: agente com menor F lidera
  - **Critério de Sucesso:** Decisão coletiva converge em <100ms
- [ ] **4.2.3** [P3] Benchmark: multi-agente vs single-agente em ambiente complexo
  - **Critério de Sucesso:** Multi-agente ≥20% melhor
- [ ] **4.2.4** [P3] Tolerância a falhas: agente desconecta, rede continua
  - **Critério de Sucesso:** Rede funciona com 1 de 3 agentes offline

---

## ═══════════════════════════════════════════════════════════
## EIXO 05 — DEPLOYMENT (WASM, ARM, PUBLICAÇÃO)
## ═══════════════════════════════════════════════════════════

### 5.1 — Edge Deployment (lab23-edge-deployment)

**Objetivo:** Compilar motor CROM para WASM e ARM.

- [ ] **5.1.0** [P2] Criar pasta `pesquisa1/labs/lab23-edge-deployment/`
- [ ] **5.1.1** [P2] Compilar pesquisa0.go para WASM com TinyGo
  - **Critério de Sucesso:** .wasm gerado, <5MB
- [ ] **5.1.2** [P2] Criar demo HTML: Agente CROM rodando no browser
  - **Critério de Sucesso:** Página funcional, <2s cold start
- [ ] **5.1.3** [P2] Benchmark WASM vs nativo: latência, memória
  - **Critério de Sucesso:** WASM ≤3x overhead vs nativo
- [ ] **5.1.4** [P3] Cross-compile para ARM (Raspberry Pi / Android)
  - **Critério de Sucesso:** Binário funcional em ARM64
- [ ] **5.1.5** [P3] Integrar com WasmEdge para aceleração GPU via WASI-NN
  - **Critério de Sucesso:** Demonstração com GPU plugin

### 5.2 — Paper arXiv (lab24-paper-arxiv)

**Objetivo:** Redigir paper acadêmico para publicação.

- [ ] **5.2.0** [P2] Criar pasta `pesquisa1/labs/lab24-paper-arxiv/`
- [ ] **5.2.1** [P2] Estruturar paper: Abstract, Introduction, Method, Results, Discussion
  - **Critério de Sucesso:** Outline de 3 páginas aprovado
- [ ] **5.2.2** [P2] Escrever seção de Resultados com dados de pesquisa0 + pesquisa1
  - **Critério de Sucesso:** Todas as tabelas e figuras prontas
- [ ] **5.2.3** [P3] Revisão e formatação LaTeX
  - **Critério de Sucesso:** PDF compilado, pronto para submissão
- [ ] **5.2.4** [P3] Submeter ao arXiv
  - **Critério de Sucesso:** Preprint publicado

---

## ═══════════════════════════════════════════════════════════
## EIXO 06 — VALIDAÇÃO CRUZADA E DOCUMENTAÇÃO
## ═══════════════════════════════════════════════════════════

### 6.1 — Validações Cruzadas

- [ ] **6.1.1** [P2] CommVQ × DeltaStore: codebook comutativo funciona com branches?
  - **Critério de Sucesso:** Branch recovery sem degradação
- [ ] **6.1.2** [P2] Detector × Streaming: ensemble funciona token-by-token?
  - **Critério de Sucesso:** Recall ≥95% em modo streaming
- [ ] **6.1.3** [P2] WASM × Multi-agente: 2 agentes WASM no browser comunicam?
  - **Critério de Sucesso:** Demo funcional com WebRTC ou WebSocket
- [ ] **6.1.4** [P2] Go motor × LLaMA: codebook Go processa KV de LLaMA?
  - **Critério de Sucesso:** Go lê codebook treinado no Colab
- [ ] **6.1.5** [P3] Ed25519 × P2P: assinaturas verificadas em rede distribuída
  - **Critério de Sucesso:** Delta rejeitado se tampered
- [ ] **6.1.6** [P3] Perplexity × branches: medir se branches pioram qualidade
  - **Critério de Sucesso:** Perplexity increase <1% após collapse

### 6.2 — Documentação Final

- [ ] **6.2.1** [P1] Atualizar README.md do repositório com pesquisa1
  - **Critério de Sucesso:** Link + resumo de resultados
- [ ] **6.2.2** [P2] Atualizar ROADMAP.md com Fase 6 (pesquisa1)
  - **Critério de Sucesso:** Items documentados
- [ ] **6.2.3** [P2] Gerar tabela cruzada de hipóteses H1-H12 com veredictos
  - **Critério de Sucesso:** CONCLUSOES.md preenchido
- [ ] **6.2.4** [P2] Referências cruzadas pesquisa0 ↔ pesquisa1
  - **Critério de Sucesso:** Links bidirecionais nos README
- [ ] **6.2.5** [P3] Publicar como release no GitHub
  - **Critério de Sucesso:** Tag + changelog + binários

---

## 📊 RESUMO QUANTITATIVO

| Eixo | Items | Labs | Foco |
|:-----|:------|:-----|:-----|
| 01 — Motor Go Nativo | 25 | Lab13, 17, 19 | Performance <1ms |
| 02 — Escala & Validação | 20 | Lab14, 16, 22 | Modelos >1B |
| 03 — Segurança & Detecção | 14 | Lab15, 20 | Zero alucinações |
| 04 — Distribuição P2P | 12 | Lab18, 21 | Multi-agente |
| 05 — Deployment | 12 | Lab23, 24 | WASM + paper |
| 06 — Cruzadas & Docs | 16 | — | Integração |
| **TOTAL** | **~99** | **12 labs** | |

### Ordem de Execução Recomendada

```
FASE 1 (P1 — Fundação):           ~30 items, ~2 semanas
├── Lab13: Agente CROM v2 em Go
├── Lab17: Codebook training Go
├── Lab19: Formato .crom v2
└── Lab14: KV Cache LLaMA-7B (Colab)

FASE 2 (P2 — Expansão):           ~45 items, ~3 semanas
├── Lab16: CommVQ + PCA
├── Lab15: Ensemble detector
├── Lab20: Streaming inference
├── Lab18: Delta P2P
├── Lab22: Benchmark suite
├── Lab23: WASM deployment
└── Lab21: Multi-agente

FASE 3 (P3 — Fronteira):          ~24 items, ~2 semanas
├── Otimizações avançadas
├── Lab24: Paper arXiv
└── Validações cruzadas finais
```

---

## 🎯 HIPÓTESES A TESTAR

| ID | Hipótese | Lab(s) | Métrica de Sucesso |
|----|----------|--------|-------------------|
| H1 | Agente CROM v2 <0.5ms/step | Lab13 | Benchmark Go vs Python |
| H2 | CommVQ comutativo com RoPE | Lab16 | Erro comutatividade <1% |
| H3 | PCA+VQ >100x compressão KV | Lab14+16 | Ratio medido no LLaMA |
| H4 | Codebook K-adaptativo via lFCI | Lab22 | K ajusta, compressão ≥10% melhor |
| H5 | EoRA recovery no DeltaStore | Lab17 | Branch recovery sem degradação |
| H6 | Ensemble detector P≥90% R≥95% | Lab15 | Métricas em 50+ test cases |
| H7 | Detecção streaming <1ms/token | Lab20 | Latência medida |
| H8 | Delta P2P funcional 3+ peers | Lab18 | Demo em LAN |
| H9 | Multi-agente consenso epistêmico | Lab21 | Convergência em <100ms |
| H10 | WASM <5MB, <2s cold start | Lab23 | Tamanho + benchmark |
| H11 | Perplexity LLaMA <2% loss | Lab14 | Perplexity medida |
| H12 | Paper aceito (arXiv) | Lab24 | Preprint publicado |

---

> **Tamanho total:** ~99 items em 12 labs, cobrindo 5 eixos + validação cruzada + documentação.
> **Estimativa de esforço:** ~2 semanas Fase 1, ~3 semanas Fase 2, ~2 semanas Fase 3.
> **Base científica:** 18 papers de fronteira (2025-2026) documentados em REFERENCIAS.md.

---

*"A pesquisa0 provou que funciona. A pesquisa1 vai fazer funcionar rápido."*
