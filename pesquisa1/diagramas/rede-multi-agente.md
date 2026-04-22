# Diagrama: Rede Multi-Agente CROM

```mermaid
graph TB
    subgraph "Agente A"
        WMA["World Model A"]
        BEA["Branches A"]
        FEA["F_A = 0.42"]
    end

    subgraph "Agente B"
        WMB["World Model B"]
        BEB["Branches B"]
        FEB["F_B = 0.38"]
    end

    subgraph "Agente C"
        WMC["World Model C"]
        BEC["Branches C"]
        FEC["F_C = 0.51"]
    end

    subgraph "Protocolo Sinapse P2P"
        MSG["DELTA_UPDATE<br/>COLLAPSE_SIGNAL<br/>FREE_ENERGY_SHARE"]
    end

    subgraph "Consenso"
        CON["Consenso Epistêmico<br/>F_global = min(F_A, F_B, F_C)<br/>= 0.38 (Agente B lidera)"]
    end

    WMA <--> MSG
    WMB <--> MSG
    WMC <--> MSG
    MSG --> CON

    style FEA fill:#f9ca24,color:#333
    style FEB fill:#4ecdc4,color:#fff
    style FEC fill:#ff6b6b,color:#fff
    style CON fill:#6c5ce7,color:#fff
```

## Protocolo

1. Cada agente calcula sua **free energy F** local
2. Via Sinapse P2P, agentes compartilham F + deltas
3. O agente com **menor F** (melhor modelo do mundo) lidera
4. Outros agentes recebem delta updates para convergir
5. **Consenso epistêmico** emerge sem servidor central
