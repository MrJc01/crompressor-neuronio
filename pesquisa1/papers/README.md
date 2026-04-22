# 📚 Papers — Pesquisa1

## Índice

| Paper | Conteúdo | Status |
|-------|----------|--------|
| — | (Papers serão gerados conforme labs forem executados) | Pendente |

## Referências Bibliográficas por Tema

### KV Cache Compression (2025-2026)
| Paper | Autores/Instituição | Ano | Relevância |
|:------|:--------------------|:----|:-----------|
| TurboQuant | Google Research | 2026 | VQ training-free, 3-bit, 8x speedup H100 |
| KVTC | ICLR | 2026 | PCA + quantização + entropy coding, 20-40x |
| CommVQ | Apple Research | 2025 | RoPE-comutativo, 1-bit KV, 32x |
| ChunkKV | NeurIPS | 2025 | Chunks semânticos, 26.5% throughput↑ |
| PyramidKV | Microsoft | 2025 | Cache adaptativo por layer |
| LogQuant | ICLR | 2025 | Log-distributed 2-bit filtering |
| NVFP4 | NVIDIA | 2025 | Hardware-native 4-bit, Blackwell GPUs |

### Delta & Weight Compression
| Paper | Autores/Instituição | Ano | Relevância |
|:------|:--------------------|:----|:-----------|
| DeltaLLM | — | 2025 | Weight sharing + low-rank deltas |
| 1-bit Delta Schemes | — | 2025 | Sign-only deltas + per-axis scaling |
| EoRA | NVIDIA | 2025 | Eigenspace recovery sem re-treino |
| Sparse Memory FT | — | 2026 | TF-IDF slot scoring, anti-forgetting |

### Dimensionalidade Intrínseca
| Paper | Autores/Instituição | Ano | Relevância |
|:------|:--------------------|:----|:-----------|
| lFCI | Fadanni et al. | 2026 | ID robusto para neural manifolds |
| SMDS | Tiblias, TMLR | 2025 | Manifold analysis em LLMs |
| GAMLA | — | 2025 | Analytical manifold learning |

### Edge / WASM
| Paper | Fonte | Ano | Relevância |
|:------|:------|:----|:-----------|
| WasmEdge | CNCF | 2026 | WASI-NN, cold start 1.5ms |
| WASI-NN Spec | W3C | 2026 | Interface padrão para HW accelerators |

### Detecção de Alucinação
| Paper | Autores | Ano | Relevância |
|:------|:--------|:----|:-----------|
| SelfCheckGPT | Manakul | 2023 | Auto-consistência sem referência |
| SAFE | Google | 2024 | Fact-checking com search engine |
| FaithEval | — | 2025 | Benchmark faithfulness multi-tarefa |

### Active Inference
| Paper | Autores | Ano | Relevância |
|:------|:--------|:----|:-----------|
| Active Inference on Discrete State-Spaces | Da Costa et al. | 2020 | Framework teórico |
| pymdp | Conti & Bhatt | 2024 | Implementação Python de referência |
