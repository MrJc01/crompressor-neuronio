# Diagrama: Pipeline de Compressão KV Cache v2

```mermaid
graph LR
    subgraph "KV Cache Original"
        KV["Key/Value Tensors<br/>(seq × heads × dim)<br/>~134MB para LLaMA-7B"]
    end

    subgraph "Pipeline Crompressor"
        PCA["PCA Decorrelation<br/>(KVTC-style)<br/>Remove redundância"]
        VQ["CommVQ Codebook<br/>(RoPE-comutativo)<br/>Quantiza para índices"]
        DELTA["Delta Store<br/>(XOR sparse)<br/>Branches em memória mínima"]
        ENT["Entropy Coding<br/>(opcional)<br/>Compressão adicional"]
    end

    subgraph "KV Cache Comprimido"
        IDX["Índices + Codebook<br/>~0.5MB estimado<br/>🎯 >100x compressão"]
    end

    KV --> PCA
    PCA --> VQ
    VQ --> DELTA
    VQ --> ENT
    DELTA --> IDX
    ENT --> IDX

    style KV fill:#ff6b6b,color:#fff
    style IDX fill:#4ecdc4,color:#fff
    style PCA fill:#45b7d1,color:#fff
    style VQ fill:#6c5ce7,color:#fff
    style DELTA fill:#96ceb4,color:#fff
```

## Comparação com SOTA

| Etapa | Paper de Referência | Resultado Isolado | Combinado (estimativa) |
|-------|--------------------|--------------------|----------------------|
| PCA decorrelation | KVTC (ICLR 2026) | 20-40x | — |
| CommVQ (RoPE) | CommVQ (Apple 2025) | 16-32x | — |
| Delta branches | **Crompressor (nós)** | 99.9% economia | — |
| **Pipeline completo** | **Ninguém** | — | **>100x** |
