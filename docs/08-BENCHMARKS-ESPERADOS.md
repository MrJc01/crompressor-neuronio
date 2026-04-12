# 📊 Benchmarks Esperados

> *"Não confiamos em intuição. Confiamos em dados."*

---

## Hipóteses Mensuráveis

### H1: Compressão do Modelo → .crom
| Métrica | Hipótese | Como Medir | Threshold de Sucesso |
|:---|:---|:---|:---|
| Compression Ratio | > 3.5x | `original_size / crom_size` | ≥ 3.0x |
| DNA Encoding Overhead | < 5% | `(dna_size - raw_compressed) / raw_compressed` | ≤ 10% |
| Merkle Tree Overhead | < 2% | `merkle_size / total_crom_size` | ≤ 5% |
| Codebook Size | < 10% do total | `codebook_size / total_crom_size` | ≤ 15% |

### H2: Tamanho do Tensor Delta
| Métrica | Hipótese | Como Medir | Threshold |
|:---|:---|:---|:---|
| Delta / Cérebro | < 5% | `delta_size / brain_crom_size` | ≤ 10% |
| Esparsificação | > 80% zeros | `count_zeros(delta) / len(delta)` | ≥ 70% |
| Delta Comprimido | < 1% | `compressed_delta / brain_crom_size` | ≤ 3% |

### H3: Performance de Inferência
| Métrica | Hipótese | Como Medir | Threshold |
|:---|:---|:---|:---|
| TTFT (Time To First Token) | < 1s (CPU) | Timestamp do primeiro token gerado | ≤ 2s |
| Latência XOR Delta | < 10ms | Benchmark ns/op | ≤ 50ms |
| Latência VQ Delta | < 50ms | Benchmark ns/op | ≤ 100ms |
| RAM Peak | < 1 GB | Go runtime.MemStats | ≤ 2 GB |
| Throughput (tokens/s) | > 5 tok/s (CPU) | Tokens gerados por segundo | ≥ 2 tok/s |

### H4: Qualidade de Saída
| Métrica | Hipótese | Como Medir | Threshold |
|:---|:---|:---|:---|
| BLEU (delta vs original) | > 0.90 | BLEU score entre saídas | ≥ 0.85 |
| Similaridade Coseno | > 0.95 | Cosine similarity dos embeddings | ≥ 0.90 |
| Perplexidade | < 1.2x baseline | PPL do modelo com delta / PPL original | ≤ 1.5x |

### H5: Termodinâmica e Entropia
| Métrica | Hipótese | Como Medir | Threshold |
|:---|:---|:---|:---|
| Shannon Entropy (brain) | Mensurável e estável | Shannon entropy dos chunks | Variância < 5% |
| Entropy (delta) | < Entropy (brain) | Shannon entropy do delta | Delta H < Brain H |
| Entropy Drift (semi-fixo) | < 10% após 100 updates | Δ Shannon após N atualizações | ≤ 15% |

### H6: Multi-Brain Routing
| Métrica | Hipótese | Como Medir | Threshold |
|:---|:---|:---|:---|
| Routing Decision Time | < 5ms | Latência do HNSW top-K | ≤ 10ms |
| 2-Brain vs 1-Brain Quality | > 5% melhoria | BLEU(2-brain) - BLEU(1-brain) | ≥ 0% |
| N-Brain Memory Overhead | < N × 100 MB | RAM com N brains montados | Linear ou melhor |

---

## Matriz de Testes

| Teste | Vertente | Dados | Script | Saída |
|:---|:---|:---|:---|:---|
| `test_compression_ratio` | Todas | Modelo GGUF 1.5B | `run_all_tests.sh` | `compression.json` |
| `test_delta_size` | Fixo | Delta gerado | `run_all_tests.sh` | `delta_metrics.json` |
| `test_xor_latency` | Fixo | Chunks variados | `benchmark.sh` | `xor_bench.json` |
| `test_vq_latency` | Semi-Fixo | Codebook + offsets | `benchmark.sh` | `vq_bench.json` |
| `test_entropy` | Todas | brain.crom | `run_all_tests.sh` | `entropy.json` |
| `test_hnsw_routing` | Dinâmico | Multi-brain | `benchmark.sh` | `routing.json` |
| `test_memory_peak` | Todas | FUSE mount | `run_all_tests.sh` | `memory.json` |
| `test_sparsity` | Fixo | Delta gerado | `run_all_tests.sh` | `sparsity.json` |

---

## Modelos Alvo para Benchmark

| Modelo | Parâmetros | Tamanho Original | Alvo .crom |
|:---|:---|:---|:---|
| Qwen2.5-0.5B | 0.5B | 1 GB | ~300 MB |
| Qwen2.5-1.5B | 1.5B | 3 GB | ~800 MB |
| LLaMA-3.2-1B | 1B | 2 GB | ~550 MB |
| Phi-3-mini-4k | 3.8B | 7.6 GB | ~2 GB |

---

## Visualização dos Resultados

Todos os dados são salvos em JSON/CSV e visualizados via:

```bash
# Dashboard interativo (Python)
cd pesquisas/visualizacao
python dashboard.py

# Gráficos estáticos (incluídos nos relatórios)
python visualizar_resultados.py
```

### Gráficos Gerados
1. **Compression Ratio** por modelo (bar chart)
2. **Delta Size vs Brain Size** (scatter plot)
3. **XOR vs VQ Latency** (box plot comparativo)
4. **Entropy Timeline** para neurônio semi-fixo (line chart)
5. **Memory Peak** por vertente (grouped bar)
6. **BLEU Score Distribution** (violin plot)
7. **Routing Decision Time** vs N neurônios (line chart)
8. **Sparsity Heatmap** do delta (heatmap)

---

> **Próximo:** [09 — Roadmap](09-ROADMAP.md)
