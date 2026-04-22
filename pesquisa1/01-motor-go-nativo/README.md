# 🔧 Eixo 01 — Motor Go Nativo: Performance <1ms

> **Pergunta Central:** Como portar o motor CROM 5D para Go nativo atingindo latência sub-milissegundo por step de inferência?

---

## Contexto

A pesquisa0 provou que o pipeline completo (Sensor→WorldModel→Branches→Decision→Firewall) funciona a 4.77ms/step em Python. O Go nativo já demonstrou viabilidade com o DeltaBranchStore (95% redução, 4.1ms/create) e Sinapse (goroutines + channels).

O objetivo é **10x speedup** sobre Python: <0.5ms/step.

## Documentos Neste Eixo

| Arquivo | Foco |
|:--------|:-----|
| [agent-crom-v2.md](agent-crom-v2.md) | Arquitetura do Agente CROM v2 em Go puro |
| [codebook-training-go.md](codebook-training-go.md) | K-Means e VQ em Go sem CGo |
| [crom-format-v2.md](crom-format-v2.md) | Formato binário .crom v2 para codebooks neurais |

## Tese Central

> O Go oferece goroutines nativas, zero-allocation paths, e compilação para WASM — o ambiente perfeito para um motor de inferência que precisa ser **soberano** (sem cloud), **rápido** (<1ms), e **portável** (browser, ARM, x86).

## Papers Relevantes

- **DeltaLLM** (2025) — Weight sharing + low-rank deltas
- **EoRA** (NVIDIA, 2025) — Eigenspace recovery sem re-treinamento
