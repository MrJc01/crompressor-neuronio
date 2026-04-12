# 🗺️ Roadmap de Pesquisa

> *"4 fases. Do neurônio congelado à rede de cognição distribuída."*

---

## Visão Geral

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   FASE 1     │───▶│   FASE 2     │───▶│   FASE 3     │───▶│   FASE 4     │
│ Brain Freeze │    │ Tensor Delta │    │  Multi-Brain │    │  P2P Soberano│
│              │    │              │    │   Routing    │    │              │
│ Congelar     │    │ XOR + VQ     │    │ Composição   │    │ Rede P2P     │
│ modelo em    │    │ sobre .crom  │    │ de neurônios │    │ de neurônios │
│ .crom        │    │              │    │              │    │              │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
   Semanas              Semanas             Semanas              Semanas
    1-4                  5-8                 9-12                 13-16
```

---

## Fase 1: Brain Freeze (Semanas 1-4)

**Objetivo:** Validar que um modelo completo pode ser congelado em `.crom` e servir como neurônio fixo funcional.

### Tarefas
| # | Tarefa | Critério de Sucesso |
|---|--------|---------------------|
| 1.1 | Comprimir modelo GGUF via Crompressor core | Ratio > 3x |
| 1.2 | Implementar flag FROZEN no header .crom | Flag persiste e é verificada |
| 1.3 | Montar .crom via FUSE e ler chunks aleatórios | Leitura O(1), < 1ms por chunk |
| 1.4 | Calcular Shannon Entropy de cada chunk | Distribuição documentada |
| 1.5 | Benchmark: compressão vs modelos alvo | Tabela com 4 modelos |
| 1.6 | Merkle Tree completa e verificável | 100% integridade |

### Testes Gerados
```bash
pesquisas/testes/cmd/test_brain_freeze/main.go
pesquisas/dados/fase1_compression.json
pesquisas/dados/fase1_entropy.json
pesquisas/relatorios/fase1_report.md
```

---

## Fase 2: Tensor Delta (Semanas 5-8)

**Objetivo:** Demonstrar que tensores delta produzem saídas adaptativas sem modificar o neurônio fixo.

### Tarefas
| # | Tarefa | Critério de Sucesso |
|---|--------|---------------------|
| 2.1 | Implementar XOR Delta sobre chunks .crom | Reversível, < 10ms |
| 2.2 | Implementar VQ Delta no espaço do codebook | Granularidade semântica verificável |
| 2.3 | Gerar deltas sintéticos para teste | 10 deltas variados |
| 2.4 | Medir esparsificação dos deltas | > 80% zeros |
| 2.5 | Benchmark: XOR vs VQ em latência | Tabela comparativa |
| 2.6 | Medir qualidade: delta vs original (BLEU) | BLEU > 0.90 |

### Testes Gerados
```bash
pesquisas/testes/cmd/test_tensor_delta/main.go
pesquisas/dados/fase2_xor_latency.json
pesquisas/dados/fase2_vq_latency.json
pesquisas/dados/fase2_sparsity.json
pesquisas/relatorios/fase2_report.md
```

---

## Fase 3: Multi-Brain Routing (Semanas 9-12)

**Objetivo:** Compor múltiplos neurônios fixos para gerar "criatividade emergente".

### Tarefas
| # | Tarefa | Critério de Sucesso |
|---|--------|---------------------|
| 3.1 | Montar N neurônios .crom simultâneos via FUSE | N ≤ 5 sem degradação |
| 3.2 | Implementar HNSW routing entre neurônios | Decision time < 5ms |
| 3.3 | Top-K seleção e composição ponderada | Output coerente |
| 3.4 | Benchmark: 1-brain vs 2-brain vs 3-brain | BLEU comparativo |
| 3.5 | Medir RAM overhead por neurônio adicional | Linear ou melhor |
| 3.6 | Detector de "colapso" multi-brain | Alerta se quality < threshold |

### Testes Gerados
```bash
pesquisas/testes/cmd/test_multi_brain/main.go
pesquisas/dados/fase3_routing.json
pesquisas/dados/fase3_composition.json
pesquisas/relatorios/fase3_report.md
```

---

## Fase 4: P2P Soberano (Semanas 13-16)

**Objetivo:** Distribuir deltas via rede P2P com segurança pós-quântica.

### Tarefas
| # | Tarefa | Critério de Sucesso |
|---|--------|---------------------|
| 4.1 | Assinar deltas com Dilithium | Signature verificável |
| 4.2 | Criptografar deltas com ChaCha20 | AEAD funcional |
| 4.3 | Transmissão via Kademlia/LibP2P | Delta recebido e aplicado |
| 4.4 | Anti-replay (nonce + timestamp) | Replay rejeitado |
| 4.5 | Merkle parcial para semi-fixo | Recálculo O(log N) |
| 4.6 | Documentação completa de deployment | README final |

### Testes Gerados
```bash
pesquisas/testes/cmd/test_p2p_delta/main.go
pesquisas/dados/fase4_security.json
pesquisas/relatorios/fase4_report.md
```

---

## Prioridades

### 🔴 P0 — Crítico (Fase 1)
- [ ] Compressão de modelo em .crom
- [ ] FUSE mount e leitura O(1)
- [ ] Shannon Entropy baseline
- [ ] Merkle Tree completa
- [ ] Scripts de teste automatizados

### 🟡 P1 — Importante (Fase 2)
- [ ] XOR Delta funcional
- [ ] VQ Delta funcional
- [ ] Benchmark comparativo
- [ ] Esparsificação medida
- [ ] Visualização de dados

### 🟢 P2 — Desejável (Fase 3)
- [ ] Multi-brain routing
- [ ] HNSW decision engine
- [ ] Composição ponderada
- [ ] Detector de colapso

### 🔵 P3 — Futuro (Fase 4)
- [ ] Dilithium signatures
- [ ] P2P delta exchange
- [ ] Blockchain de cognição
- [ ] Deployment guide

---

## Métricas de Sucesso por Fase

| Fase | Métrica | Meta |
|------|---------|------|
| 1 | Compression Ratio | > 3x |
| 1 | FUSE read latency | < 1ms |
| 2 | Delta / Brain size | < 5% |
| 2 | XOR latency | < 10ms |
| 3 | Routing decision | < 5ms |
| 3 | Multi-brain BLEU | > single-brain BLEU |
| 4 | Signature verify | < 1ms |
| 4 | P2P round-trip | < 500ms |

---

> **Próximo:** [10 — Glossário](10-GLOSSARIO.md)
