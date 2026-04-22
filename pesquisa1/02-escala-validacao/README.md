# 🔬 Eixo 02 — Escala e Validação: Modelos >1B Params

> **Pergunta Central:** Os resultados da pesquisa0 (GPT-2, 124M) escalam para modelos de produção (LLaMA-7B+)?

---

## Contexto

A pesquisa0 validou KV Cache compression no GPT-2 (94.2% redução). Mas modelos reais têm 7B-70B params. Os papers de 2025-2026 sugerem que compressão **escala super-linearmente** — quanto maior o modelo, melhor o ratio.

O risco R2 da pesquisa0 ("KV Cache não escala >1B") precisa ser resolvido.

## Documentos Neste Eixo

| Arquivo | Foco |
|:--------|:-----|
| [kv-cache-llama.md](kv-cache-llama.md) | Validação com LLaMA-7B real (perplexity) |
| [codebook-rope.md](codebook-rope.md) | Codebook comutativo com RoPE (CommVQ) |
| [benchmark-suite.md](benchmark-suite.md) | Suite de benchmarks vs SOTA (TurboQuant, KVTC) |

## Tese Central

> Se CommVQ + PCA decorrelation + nosso Delta Storage combinam, o Crompressor pode atingir **>100x compressão** de KV Cache com <1% perplexity loss — superando TurboQuant (6x) e KVTC (20-40x) individuamente.

## Papers Relevantes

- **TurboQuant** (Google, 2026) — 3-bit, 6x memória, 8x speedup
- **KVTC** (ICLR 2026) — PCA + entropy coding, 20-40x
- **CommVQ** (Apple, 2025) — RoPE-comutativo, 1-bit KV
- **LLM-KICK** (Apple, 2025) — Benchmark para modelos comprimidos
