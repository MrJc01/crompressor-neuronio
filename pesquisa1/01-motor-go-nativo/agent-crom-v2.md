# Agente CROM v2 — Arquitetura Go

## Visão Geral

O Agente CROM v2 é a evolução do protótipo Python (pesquisa0, blitz_final.py) para um motor Go completo com todas as garantias de produção.

## Arquitetura

```
┌──────────────────────────────────────────────────────────────┐
│                    AGENTE CROM v2 (Go)                        │
│                                                              │
│  ┌─────────┐   ┌──────────┐   ┌────────────┐   ┌─────────┐ │
│  │ Sensor  │──▶│ WorldMdl │──▶│ BranchMgr  │──▶│Decision │ │
│  │ (chan)  │   │ (EMA/KF) │   │(DeltaStore)│   │(Weighted│ │
│  └─────────┘   └──────────┘   └────────────┘   └─────────┘ │
│       ▲                            │                  │      │
│       │                       ┌────▼────┐        ┌────▼────┐ │
│       │                       │ Sinapse │        │Firewall │ │
│       │                       │(gorout.)│        │(Ed25519)│ │
│       │                       └─────────┘        └─────────┘ │
│       └──────────────── feedback loop ──────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

## Interfaces Go

```go
type Agent interface {
    Step(observation []float64) (action []float64, err error)
    GetFreeEnergy() float64
    GetBranches() []Branch
}

type WorldModel interface {
    Predict(state, action []float64) (nextState []float64)
    Update(observation []float64)
    Error() float64
}

type BranchManager interface {
    Explore(state []float64, depth int) []Branch
    Collapse(criterion func(Branch) float64) Branch
    Store() *DeltaBranchStore
}

type Firewall interface {
    Check(prediction []float64) (safe bool, confidence float64)
    Sign(data []byte) ([]byte, error)
}
```

## Metas de Performance

| Métrica | Python (pesquisa0) | Go v2 (meta) | Speedup |
|---------|-------------------|--------------|---------|
| Step latência | 4.77ms | **<0.5ms** | 10x |
| Branch create | 4.1ms | **<0.3ms** | 14x |
| Sinapse msg | ~100μs | **<10μs** | 10x |
| Memória/branch | ~50KB | **<5KB** | 10x |
| Sign delta | 122μs | **<50μs** | 2.5x |
