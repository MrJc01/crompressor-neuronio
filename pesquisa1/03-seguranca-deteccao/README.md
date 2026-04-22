# 🛡️ Eixo 03 — Segurança e Detecção: Zero Alucinações

> **Pergunta Central:** Como construir um detector de alucinação que opera em tempo real durante streaming inference, usando o próprio codebook como base?

---

## Contexto

A pesquisa0 evoluiu o detector de v1 (recall 68%) para v3 (recall 100%). O próximo passo é:
1. **Ensemble v1+v3** para maximizar precision E recall
2. **Detecção em tempo real** durante inference (não pós-processamento)
3. **Zero-model detector** — usar o codebook, não um modelo auxiliar

## Documentos Neste Eixo

| Arquivo | Foco |
|:--------|:-----|
| [ensemble-detector.md](ensemble-detector.md) | Combinação v1+v3 com voting scheme |
| [streaming-detection.md](streaming-detection.md) | Detecção durante inference, não após |
| [codebook-as-detector.md](codebook-as-detector.md) | O codebook como firewall integrado |

## Tese Central

> O codebook usado para compressão do KV Cache contém implicitamente um **modelo do que é "normal"**. Tokens cuja distância ao centróide mais próximo excede um limiar são, por definição, **fora da distribuição** — candidatos a alucinação. Compressão e detecção são o mesmo componente.

## Papers Relevantes

- **SelfCheckGPT** (Manakul, 2023) — Auto-consistência
- **SAFE** (Google, 2024) — Fact-checking com search
- **FaithEval** (2025) — Benchmark multi-tarefa
