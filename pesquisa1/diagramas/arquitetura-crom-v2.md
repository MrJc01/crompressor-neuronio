# Diagrama: Motor CROM v2 — Arquitetura Completa

```mermaid
graph TB
    subgraph "ENTRADA"
        OBS["📡 Observações<br/>Sensores / Dados"]
    end

    subgraph "MOTOR CROM v2 (Go Nativo)"
        subgraph "CAMADA 1 — Percepção"
            SENS["Sensor<br/>🔬 Normalização + Ruído"]
            WM["World Model<br/>🧠 EMA/Kalman Filter"]
        end

        subgraph "CAMADA 2 — Simulação (5D)"
            BE["Branch Engine<br/>🌿 DeltaBranchStore"]
            CB["Codebook<br/>📦 CommVQ + PCA"]
            MCTS["MCTS Explorer<br/>🎯 Active Inference"]

            B1["Branch 1<br/>(delta)"]
            B2["Branch 2<br/>(delta)"]
            BN["Branch N<br/>(delta)"]
        end

        subgraph "CAMADA 3 — Decisão"
            DEC["Decision<br/>⚖️ Weighted Collapse"]
            FW["Firewall<br/>🛡️ Codebook Distance"]
            SIG["Ed25519<br/>🔐 Sign & Verify"]
        end

        subgraph "CAMADA 4 — Comunicação"
            SYN["Sinapse P2P<br/>🔗 goroutines+channels"]
        end
    end

    subgraph "SAÍDA"
        ACT["🎬 Ação / Output"]
        OTHER["🌐 Outros Agentes"]
    end

    OBS --> SENS
    SENS --> WM
    WM --> BE
    BE --> CB
    CB --> MCTS

    MCTS --> B1
    MCTS --> B2
    MCTS --> BN

    B1 --> DEC
    B2 --> DEC
    BN --> DEC

    DEC --> FW
    FW --> SIG
    SIG --> ACT

    SYN <--> BE
    SYN <--> OTHER

    style WM fill:#4ecdc4,color:#fff
    style CB fill:#6c5ce7,color:#fff
    style FW fill:#ff6b6b,color:#fff
    style SYN fill:#f9ca24,color:#333
    style MCTS fill:#45b7d1,color:#fff
    style BE fill:#96ceb4,color:#fff
```

## Fluxo

1. **Observação** entra no sensor → normalizada
2. **World Model** atualiza predição interna
3. **Branch Engine** gera N futuros possíveis (deltas XOR)
4. **Codebook** comprime cada branch (CommVQ, RoPE-comutativo)
5. **MCTS** explora a árvore com Active Inference (minimiza free energy)
6. **Decision** colapsa na branch ótima (weighted por variância)
7. **Firewall** verifica distância ao codebook (alucinação?)
8. **Ed25519** assina o output (soberania)
9. **Sinapse** distribui deltas para outros agentes via P2P
