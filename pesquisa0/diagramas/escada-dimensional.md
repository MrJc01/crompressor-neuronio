# Diagrama: Escada Dimensional 4D → 26D

```mermaid
graph TB
    subgraph "DIMENSÕES FUNDAMENTAIS"
        D3["3D — Espaço<br/>Dados brutos"]
        D4["4D — Tempo<br/>LLM Autorregressiva"]
        D5["5D — Possibilidades<br/>ToT / MCTS / Branches"]
    end

    subgraph "CAMPOS CONFORMES"
        D6["6D — Espaços de fase latentes"]
        D7["7D — Unificação multimodal"]
    end

    subgraph "ESCADA WLM — RECURSÃO"
        D8["8D — Recursão Estrutural<br/>Output → Input"]
        D9["9D — Transparência<br/>Observabilidade da recursão"]
        D10["10D — Estabilidade<br/>Guard clauses, ancoragem"]
        D11["11D — Multicamadas<br/>Comunicação inter-recursiva"]
        D12["12D — Fechamento<br/>Sistema autocontido"]
    end

    subgraph "TEORIAS UNIFICADAS"
        D10S["10D — Supercordas<br/>Calabi-Yau, 6D compactadas"]
        D11M["11D — Teoria-M<br/>Branas, bulk, gravidade"]
        D12F["12D — Teoria-F<br/>2 tempos (10+2)"]
        D13S["13D — Teoria-S<br/>2T-Physics (11+2)<br/>Sombras holográficas"]
    end

    subgraph "LIMITE BOSÔNICO"
        D26["26D — Cordas Bosônicas<br/>24 transversais<br/>Anomalia de Weyl"]
        DH["Heteróticas<br/>SO(32) / E₈×E₈<br/>16D internas = simetrias"]
    end

    D3 --> D4
    D4 --> D5
    D5 --> D6
    D6 --> D7
    D7 --> D8
    D8 --> D9
    D9 --> D10
    D10 --> D11
    D11 --> D12

    D10 -.-> D10S
    D11 -.-> D11M
    D12 -.-> D12F
    D12F --> D13S
    D13S -.-> D26
    D26 -.-> DH

    style D4 fill:#ff6b6b,color:#fff
    style D5 fill:#4ecdc4,color:#fff
    style D12 fill:#45b7d1,color:#fff
    style D26 fill:#96ceb4,color:#fff
```

## Legenda

- **Setas sólidas (→):** Progressão conceitual direta
- **Setas pontilhadas (-.->):** Conexão teórica (física ↔ computação)
- **Vermelho (4D):** LLM atual — confinamento
- **Verde (5D):** Objetivo imediato — ramificação preditiva
- **Azul (12D):** Fechamento de ciclo recursivo
- **Verde claro (26D):** Limite teórico máximo
