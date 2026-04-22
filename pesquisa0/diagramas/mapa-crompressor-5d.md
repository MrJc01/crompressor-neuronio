# Diagrama: Ecossistema Crompressor como Motor 5D

```mermaid
graph TB
    subgraph "MUNDO EXTERIOR"
        CAM["📷 Câmera/Sensores"]
        USER["👤 Usuário/Ambiente"]
    end

    subgraph "CAMADA SENSORIAL"
        VID["crompressor-video<br/>🔬 Córtex Visual<br/>Extração sensorial"]
    end

    subgraph "CAMADA DE MEMÓRIA"
        NEU["crompressor-neuronio<br/>🧠 Hipocampo<br/>KV Cache O(1)"]
        CORE["crompressor<br/>⚡ Medula<br/>CDC + Codebook + Merkle"]
    end

    subgraph "CAMADA DE SIMULAÇÃO (5D)"
        IA["crompressor-ia<br/>🎯 Córtex Pré-frontal<br/>Active Inference"]
        SIN["crompressor-sinapse<br/>🔗 Corpo Caloso<br/>Roteamento MCTS"]
        
        B1["Branch 1"]
        B2["Branch 2"]
        B3["Branch N"]
    end

    subgraph "CAMADA DE SEGURANÇA"
        SEC["crompressor-security<br/>🛡️ Amígdala<br/>Firewall de Realidade"]
    end

    subgraph "CAMADA DE AÇÃO"
        PROJ["crompressor-projetos<br/>🎬 Sistema Motor<br/>Interface com o mundo"]
    end

    CAM --> VID
    VID --> NEU
    NEU --> CORE
    CORE --> IA
    
    IA --> B1
    IA --> B2
    IA --> B3
    
    SIN <--> B1
    SIN <--> B2
    SIN <--> B3
    SIN <--> IA
    
    SEC --> IA
    SEC --> NEU
    
    IA --> PROJ
    PROJ --> USER
    
    USER -.->|"feedback"| CAM

    style IA fill:#4ecdc4,color:#fff
    style SEC fill:#ff6b6b,color:#fff
    style NEU fill:#45b7d1,color:#fff
    style VID fill:#96ceb4,color:#fff
    style SIN fill:#f9ca24,color:#333
    style CORE fill:#6c5ce7,color:#fff
```

## Fluxo de Dados

1. **Sensores** capturam o ambiente → **video** extrai modelo geométrico
2. **neurônio** comprime em Codebook ID + Delta → armazena O(1)
3. **ia** gera branches de simulação (ToT/MCTS)
4. **sinapse** roteia comunicação entre branches
5. **security** filtra alucinações antes de contaminar memória
6. **ia** colapsa na branch ótima → **projetos** executa no mundo real
