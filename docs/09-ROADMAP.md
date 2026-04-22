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

## Fase 5: Pesquisa0 — Motor 5D de Inferência Ativa `[83% COMPLETO]`

> **Adicionado em:** 2026-04-22 — Resultados da pesquisa0 integrados ao roadmap.

**Objetivo:** Validar experimentalmente o uso do Crompressor como motor de simulação multiverso para agentes com inferência ativa.

### Resultados Validados (12/12 Labs)

| Métrica | Valor | Lab |
|---------|-------|-----|
| Delta Storage economia | 99.9% | Lab07 |
| KV Cache compressão (GPT-2 real) | 94.2% (17.1x) | Lab06 |
| Active Inference speedup | 12.7x | Lab10 |
| Tree of Thoughts ganho | 2350% | Lab05 |
| Dimensionalidade efetiva | ~19D estável | Lab04 |
| Merge ponderado SNR | +9.82 dB | Lab11 |
| Dual Clock v2 melhoria | -8.7% erro | Lab12v2 |
| Sinapse bandwidth redução | 95.5% | Blitz |

### Próximas Tarefas (21 itens pendentes)

| Trilha | Items | Prioridade |
|--------|-------|------------|
| Migração Go (Delta Store, Sinapse) | 7 | 🔴 P0 |
| Validação GPU (LLaMA-7B, embeddings) | 5 | 🟡 P1 |
| Agente CROM v1 integrado | 2 | 🟡 P1 |
| Documentação e publicação | 4 | 🟢 P2 |

### Papers Gerados
- `pesquisa0/papers/papel0.md` — 8 primeiros labs
- `pesquisa0/papers/papel1.md` — 12/12 labs consolidados
- `pesquisa0/papers/papel2.md` — GPU validation + H13 corrigida
- `pesquisa0/papers/papel3.md` — Blitz final (31 itens, 83%)

---

## Métricas de Sucesso por Fase

| Fase | Métrica | Meta | Status |
|------|---------|------|--------|
| 1 | Compression Ratio | > 3x | ✅ 40.8x (tensor-vivo) |
| 1 | FUSE read latency | < 1ms | ✅ Funcional |
| 2 | Delta / Brain size | < 5% | ✅ 0.1% (Lab07) |
| 2 | XOR latency | < 10ms | ✅ 93μs (Lab09) |
| 3 | Routing decision | < 5ms | ⏳ Pendente |
| 3 | Multi-brain BLEU | > single-brain BLEU | ⏳ Pendente |
| 4 | Signature verify | < 1ms | ✅ Ed25519 |
| 4 | P2P round-trip | < 500ms | ⏳ Pendente |
| 5 | KV Cache compressão real | > 90% | ✅ 94.2% (GPT-2) |
| 5 | Active Inference speedup | > 5x | ✅ 12.7x |
| 5 | Hipóteses confirmadas | > 80% | ✅ 93% (13/14) |

---

> **Próximo:** [10 — Glossário](10-GLOSSARIO.md)
