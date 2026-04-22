# Pesquisa1 — Do Conceito ao Motor: Engenharia do CROM 5D

> **"A pesquisa0 provou que funciona. A pesquisa1 vai fazer funcionar rápido."**

## Visão Geral

A Pesquisa1 é a transição da fase de **validação experimental** (pesquisa0) para **engenharia de produção**. O objetivo é construir um motor CROM 5D funcional em Go nativo, validado em escala com modelos reais (>1B params), e pronto para integração com o ecossistema Crompressor.

## Pré-requisitos da Pesquisa0

| Hipótese | Status | Base para Pesquisa1 |
|----------|--------|---------------------|
| Delta Storage 99.9% | ✅ | Motor de branches em Go |
| KV Cache 94.2% real | ✅ | Compressão de contexto |
| Active Inference 12.7x | ✅ | Agente autônomo |
| Detector Recall 100% | ✅ | Firewall de segurança |
| Dim intrínseca 27.6D | ✅ | Codebook otimizado |
| Go nativo 4/4 tests | ✅ | Base de código Go |

## Estrutura

```
pesquisa1/
├── README.md                           # ← Este arquivo
├── PLANEJAMENTO.md                     # Checklist extenso (~150 items)
├── CONCLUSOES.md                       # Veredictos (preenchido ao final)
├── labs/
│   ├── lab13-agent-crom-v2/            Motor CROM v2 em Go (<1ms/step)
│   ├── lab14-kv-cache-llama/           KV Cache com LLaMA-7B real
│   ├── lab15-ensemble-detector/        Detector ensemble (P≥90% R≥95%)
│   ├── lab16-codebook-rope/            Codebook position-aware (RoPE)
│   ├── lab17-codebook-training-go/     Loop de treinamento em Go
│   ├── lab18-delta-p2p/                Distribuição P2P de deltas
│   ├── lab19-crom-format-v2/           Formato binário .crom v2
│   ├── lab20-streaming-inference/      Inferência com cache comprimido
│   ├── lab21-multi-agent-crom/         Multi-agente via Sinapse
│   ├── lab22-benchmark-suite/          Benchmarks vs SOTA
│   ├── lab23-edge-deployment/          Deploy ARM/WASM
│   └── lab24-paper-arxiv/              Draft para publicação
├── papers/
│   └── README.md
├── resultados/                         # JSONs de cada lab
└── diagramas/                          # Diagramas de arquitetura
```

## Eixos de Pesquisa

| # | Eixo | Labs | Foco |
|---|------|------|------|
| 1 | Motor Go Nativo | 13, 17, 19 | Performance <1ms |
| 2 | Escala & Validação | 14, 16, 22 | Modelos >1B params |
| 3 | Segurança & Detecção | 15, 20 | Produção zero-alucinação |
| 4 | Distribuição | 18, 21 | P2P e multi-agente |
| 5 | Deployment | 23, 24 | Edge + publicação |

## Links

- [Pesquisa0 (concluída)](../pesquisa0/CONCLUSOES.md) — 129/129 items, 15/16 hipóteses
- [Código Go existente](../pesquisas/testes/pkg/pesquisa0/) — Delta Store + Sinapse + Ed25519
- [Planejamento detalhado](PLANEJAMENTO.md) — Checklist extenso com critérios de sucesso
