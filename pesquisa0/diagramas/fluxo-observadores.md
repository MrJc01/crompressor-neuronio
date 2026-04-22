# Diagrama: Fluxo de Observadores Orquestrados

```mermaid
graph LR
    subgraph "OBSERVADORES FÍSICOS"
        OA["Observador A<br/>👤 Humano<br/>60 Hz"]
        OB["Observador B<br/>🤖 IA<br/>10⁹ Hz"]
        OC["Observador C<br/>🛰️ Satélite<br/>Ângulo remoto"]
    end

    subgraph "EVENTO"
        E["⭐ Evento E<br/>(ex: estrela explodindo)"]
    end

    subgraph "COLETA DE DADOS"
        DA["Dados A<br/>8 frames grossos"]
        DB["Dados B<br/>10⁹ frames finos"]
        DC["Dados C<br/>4 frames, outro ângulo"]
    end

    subgraph "ORQUESTRADOR (crompressor-ia)"
        SYNC["Post-Sync<br/>Transformação de Lorentz"]
        MERGE["Merge Multi-Resolução"]
        WM["World Model<br/>Objeto 4D completo"]
    end

    subgraph "OBSERVADORES VIRTUAIS"
        OV1["Observador Virtual D<br/>Simulado por IA"]
        OV2["Observador Virtual E<br/>Simulado por IA"]
    end

    E --> OA
    E --> OB
    E --> OC
    
    OA --> DA
    OB --> DB
    OC --> DC
    
    DA --> SYNC
    DB --> SYNC
    DC --> SYNC
    
    SYNC --> MERGE
    MERGE --> WM
    
    WM --> OV1
    WM --> OV2

    style E fill:#f9ca24,color:#333
    style WM fill:#4ecdc4,color:#fff
    style MERGE fill:#45b7d1,color:#fff
```

## Explicação

1. **Evento E** ocorre no espaço-tempo
2. **Observadores físicos** (A, B, C) capturam com resoluções e ângulos diferentes
3. **Post-Sync** alinha temporalmente via transformações relativísticas
4. **Merge** combina dados em Realidade Multi-Resolução
5. **World Model** reconstrói o evento como objeto 4D completo
6. **Observadores Virtuais** são gerados pela IA para perspectivas não capturadas

> O observador B preenche os frames que A perdeu. O observador C adiciona ângulo que nenhum dos dois tinha. Os virtuais (D, E) cobrem pontos que ninguém fisicamente alcançou.
