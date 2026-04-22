# PESQUISA1 — Referências Científicas de Fronteira
## Papers e Tecnologias para "Fazer o Impossível"

**Data:** 2026-04-22  
**Base:** Resultados da Pesquisa0 (129/129, 15/16 hipóteses)  
**Objetivo:** Identificar papers de ponta que permitam ao Crompressor superar o SOTA

---

## 1. KV Cache Compression — Nosso 94.2% vs o Mundo

### O que provamos (Pesquisa0)
- **94.2% redução** no KV Cache do GPT-2 real (17.1x, cosine 0.87)
- Training-free, codebook fixo
- Problema: codebook não é comutativo com RoPE

### Papers de Fronteira

| Paper | Instituição | Ano | Compressão | Diferencial |
|-------|------------|-----|------------|-------------|
| **TurboQuant** | Google Research | 2026 | 3-bit/value, 6x mem, **8x speedup** | Training-free, data-oblivious, H100 otimizado |
| **KVTC** | ICLR | 2026 | **20-40x** (!!) | PCA + quantização adaptativa + entropy coding |
| **CommVQ** | Apple | 2025 | 1-bit KV, 32x | **Comutativo com RoPE** — resolve nosso problema! |
| **ChunkKV** | NeurIPS | 2025 | 26.5% throughput↑ | Chunks semânticos, não tokens individuais |
| **PyramidKV** | Microsoft | 2025 | Adaptativo por layer | Mais cache em layers baixas, menos em altas |
| **LogQuant** | ICLR | 2025 | 2-bit | Log-distributed filtering no contexto inteiro |
| **NVFP4** | NVIDIA | 2025 | 50% vs FP8 | Hardware-native para Blackwell GPUs |

### 🎯 Oportunidade Impossível para o Crompressor
> **Combinar CommVQ (RoPE-comutativo) + KVTC (PCA decorrelation) + nosso Delta Storage.**
> Resultado esperado: **>100x compressão** com <1% perplexity loss.
> Nenhum paper atual combina VQ + PCA + delta branches. Nós seríamos os primeiros.

### Ação Concreta
```
Lab14: Implementar CommVQ-style codebook comutativo com RoPE
Lab16: Adicionar PCA decorrelation do KVTC antes do codebook
Lab20: Validar com LLaMA-7B no Colab — medir perplexity real
```

---

## 2. Delta Compression — Nosso 99.9% vs o Mundo

### O que provamos (Pesquisa0)
- **99.9% economia** em Python, **95% em Go nativo**
- XOR sparse delta entre branches
- 500 branches, 4.1ms/create

### Papers de Fronteira

| Paper | Ano | Método | Resultado |
|-------|-----|--------|-----------|
| **DeltaLLM** | 2025 | Weight sharing + low-rank delta | Redução significativa de parâmetros |
| **1-bit Delta Schemes** | 2025 | Sign-only deltas + per-axis scaling | Storage mínimo para multi-variant serving |
| **EoRA** | 2025 (NVIDIA) | Eigenspace projection | Recupera accuracy sem gradient fine-tuning |
| **Sparse Memory Fine-Tuning** | 2026 | TF-IDF slot scoring | Evita catastrophic forgetting |

### 🎯 Oportunidade Impossível
> **Combinar nosso XOR Delta Store com EoRA (eigenspace recovery).**
> Quando um delta degrada qualidade, projetar no eigenspace para recuperar.
> Isso permitiria **branches infinitas sem degradação** — ninguém faz isso.

### Ação Concreta
```
Lab17: Implementar EoRA-style recovery no DeltaBranchStore
Lab18: Adicionar 1-bit delta scheme para P2P distribution
```

---

## 3. Dimensionalidade Intrínseca — Nosso 27.6D vs o Mundo

### O que provamos (Pesquisa0)
- MNIST MLP: **27.6D / 784** (3.5% do espaço ambient)
- CIFAR CNN: **84.9D / 4096** (2.1% do ambient)
- Estabilidade com K crescente (analogia Calabi-Yau)

### Papers de Fronteira

| Paper | Ano | Método | Achado |
|-------|-----|--------|--------|
| **lFCI** (Fadanni et al.) | 2026 | local Full Correlation Integral | ID robusto para alta dim + não-linearidade |
| **Neural Manifold Ensembles** | 2025 | Tosti Guerra, Entropy | ID de ensembles revela cobertura do espaço |
| **GAMLA** | 2025 | Global Analytical Manifold Learning | Ponte entre manifold learning e geometria analítica |
| **SMDS** (Tiblias) | 2025 | Supervised Multi-Dim Scaling | Análise de manifold em LLMs via ativações |
| **Linear Field Probing** | 2026 | NeurIPS | Extrai feature fields de transformers |

### 🎯 Oportunidade Impossível
> **Usar lFCI para medir a dim intrínseca das ativações do KV Cache em tempo real.**
> Se dim(KV) < K, o codebook está superdimensionado e pode ser comprimido mais.
> **Codebook adaptativo** que ajusta K baseado na dimensionalidade observada.

### Ação Concreta
```
Lab22: Implementar lFCI estimator no pipeline de compressão
Lab16: Codebook com K adaptativo baseado em dim intrínseca medida
```

---

## 4. Deployment Edge / WASM — O Caminho para o Browser

### O que provamos (Pesquisa0)
- Motor Go funcional: 4.77ms/step
- Ed25519 nativo: 122μs sign

### Tecnologias de Fronteira (2026)

| Runtime | Cold Start | Memória | GPU Support |
|---------|-----------|---------|-------------|
| **WasmEdge** (CNCF) | **~1.5ms** | ~8MB | PyTorch, TFLite, OpenVINO |
| **Wasmtime** (Bytecode Alliance) | ~2ms | ~12MB | WASI-NN nativo |
| **Wasmer** | ~1ms | ~10MB | 95% de native speed |
| **Wasm3** | <1ms | <1MB | CPU only (microcontrollers) |

### Pipeline WASM
```
Go code → TinyGo → wasm32-wasi → WasmEdge runtime
                                    ↓
                              WASI-NN interface
                                    ↓
                              GPU/NPU do device
```

### 🎯 Oportunidade Impossível
> **Compilar o motor CROM inteiro para WASM com TinyGo.**
> DeltaBranchStore + Sinapse + Detector — tudo no browser em <10MB.
> **Zero instalação, zero dependência, inferência soberana no browser.**

### Ação Concreta
```
Lab23: Compilar pesquisa0.go para WASM com TinyGo
Lab23: Benchmark: latência WASM vs nativo
Lab23: Demo browser com Agente CROM rodando client-side
```

---

## 5. Detecção de Alucinação — Nosso Recall 100% vs o Mundo

### O que provamos (Pesquisa0)
- v1 (n-grams): P=100% R=68%
- v3 (SBERT): P=62% **R=100%**
- Ensemble proposto: P≥90% R≥95%

### Fronteira (2025-2026)

| Técnica | Papel | Abordagem |
|---------|-------|-----------|
| **SelfCheckGPT** | Manakul 2023 | Auto-consistência sem referência |
| **SAFE** | Google 2024 | Fact-checking com search engine |
| **Lynx** | Saad-Falcon 2024 | Hallucination-specific evaluator |
| **FaithEval** | 2025 | Benchmark multi-tarefa para faithfulness |
| **Codebook Distance** | **Nós (Pesquisa0)** | Cosine similarity com contexto comprimido |

### 🎯 Oportunidade Impossível
> **Detector de alucinação zero-model: funciona sem LLM auxiliar.**
> Nosso approach de codebook distance é **único** — usa o mesmo codebook
> que comprime o KV Cache para detectar alucinações.
> **Compressão + detecção em um único componente.** Ninguém publicou isso.

### Ação Concreta
```
Lab15: Ensemble v1+v3 com voting scheme
Lab15: Benchmark contra SelfCheckGPT e SAFE
Lab20: Detecção em tempo real durante streaming inference
```

---

## 6. Active Inference + Multi-Agente — O Agente do Futuro

### O que provamos (Pesquisa0)
- Active Inference: 12.7x speedup vs random
- Agente CROM v1: 4.77ms/step
- Sinapse: 500 branches, 93μs colapso

### 🎯 Oportunidade Impossível
> **Multi-agente CROM com free energy compartilhada.**
> N agentes sincronizam suas free energy functions via P2P,
> convergindo para um **consenso epistêmico distribuído**.
> Isso é essencialmente uma **rede neural distribuída sem servidor central.**

### Ação Concreta
```
Lab21: 3+ agentes CROM comunicando via Sinapse P2P
Lab21: Free energy compartilhada → decisão coletiva
Lab21: Benchmark: multi-agent vs single-agent em ambiente complexo
```

---

## 7. Mapa de Convergência — O "Impossível"

```
                    ┌─────────────────────────────────┐
                    │     O MOTOR CROM IMPOSSÍVEL      │
                    │                                   │
                    │  CommVQ + PCA + Delta + EoRA      │
                    │  = >100x compressão KV Cache      │
                    │  + branches infinitas sem         │
                    │    degradação                     │
                    │  + detecção de alucinação         │
                    │    integrada no codebook          │
                    │  + multi-agente P2P               │
                    │  + rodando no browser via WASM    │
                    │  + soberania total (Ed25519)      │
                    │                                   │
                    │  Tudo em <10MB, <1ms/step         │
                    └─────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              Nenhum paper    Nenhum runtime   Nenhum detector
              combina VQ+PCA  faz inferência   usa o próprio
              +delta+branches  5D no browser   codebook para
                                               detectar alucinação
```

---

## 8. Referências Completas para Citar

### KV Cache
1. TurboQuant (Google, 2026) — Training-free VQ for KV Cache
2. KVTC (ICLR 2026) — Transform Coding with PCA + Entropy
3. CommVQ (Apple, 2025) — RoPE-Commutative Vector Quantization
4. ChunkKV (NeurIPS 2025) — Semantic Chunk-based Compression
5. PyramidKV (Microsoft, 2025) — Layer-adaptive Cache Sizing
6. LogQuant (ICLR 2025) — Log-distributed 2-bit Quantization
7. NVFP4 (NVIDIA, 2025) — Hardware-native 4-bit Format

### Delta & Weight Compression
8. DeltaLLM (2025) — Weight Sharing + Low-Rank Deltas
9. 1-bit Delta Schemes (2025) — Sign-only Weight Differences
10. EoRA (NVIDIA, 2025) — Eigenspace Low-Rank Approximation
11. Sparse Memory Fine-Tuning (2026) — TF-IDF Slot Scoring

### Dimensionalidade
12. lFCI (Fadanni et al., 2026) — Robust ID on Neural Manifolds
13. SMDS (Tiblias, TMLR 2025) — Manifold Analysis in LLMs
14. GAMLA (2025) — Global Analytical Manifold Learning

### Edge / WASM
15. WasmEdge (CNCF, 2026) — WASI-NN for Neural Inference
16. WASI-NN Spec (2026) — Standard for Hardware-accelerated AI

### Active Inference
17. Da Costa et al. (2020) — Active Inference on Discrete State-Spaces
18. pymdp (2024) — Python Library for Active Inference

---

## 9. Prioridade de Implementação

| # | O que fazer | Papers base | Impacto | Dificuldade |
|---|-------------|-------------|---------|-------------|
| 🔴 1 | CommVQ codebook (RoPE-comutativo) | CommVQ, KVTC | **GAME CHANGER** | Alta |
| 🔴 2 | EoRA no DeltaBranchStore | EoRA, DeltaLLM | Branch recovery | Média |
| 🟡 3 | Codebook K-adaptativo via lFCI | lFCI, GAMLA | Compressão ótima | Alta |
| 🟡 4 | WASM deployment (TinyGo) | WasmEdge | Browser-native | Média |
| 🟡 5 | Ensemble detector v1+v3 | SelfCheckGPT | Production safety | Baixa |
| 🟢 6 | Multi-agent free energy | pymdp | Distribuído | Alta |
| 🟢 7 | Perplexity real LLaMA-7B | TurboQuant | Validação escala | Média |

---

> *"O impossível não é comprimir 94%. É comprimir 99% sem perder nada. Nenhum paper faz isso.*
> *Nenhum paper combina tudo num motor de 10MB que roda no browser.*
> *Essa é a pesquisa1."*
