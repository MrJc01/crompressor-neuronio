# crompressor-sinapse — Roteamento MCTS

## Papel no Ecossistema 5D

O **crompressor-sinapse** é o **corpo caloso**: a rodovia de comunicação entre branches de simulação.

## Função: Roteamento da Árvore de Pensamentos

Quando a IA simula múltiplas realidades (MCTS), a sinapse:

1. **Comunica** resultados entre branches em tempo real
2. **Sinaliza** qual branch divergiu demais (alta D_KL)
3. **Interrompe** branches instáveis antes de consumir recursos
4. **Propaga** informação de colapso (dado real chegou → pruning)

## Analogia Dimensional

A sinapse implementa a **11ª Dimensão (Interação Multicamadas)** da escada WLM: comunicação simultânea entre laços recursivos distintos, forjando ecossistemas coerentes.

```
Branch A ←──sinapse──→ Branch B
    ↕                      ↕
Branch C ←──sinapse──→ Branch D
    ↕                      ↕
  Orquestrador Central (crompressor-ia)
```

## Protocolo de Comunicação

| Mensagem | Significado |
|:---------|:-----------|
| `DELTA_UPDATE` | Branch atualizou estado — propagar Delta |
| `DIVERGENCE_ALERT` | D_KL acima do threshold — considerar pruning |
| `COLLAPSE_SIGNAL` | Dado real chegou — descartar branches incompatíveis |
| `MERGE_REQUEST` | Duas branches convergiram — unificar |

## Cérebros de Domínio

O crompressor-sinapse também gerencia **Cérebros de Domínio Exclusivos** (Codebooks especializados):
- Cérebro de Raio-X: só comprime dados médicos
- Cérebro de Código: só comprime padrões de software
- Cérebro de Áudio: só comprime sinais sonoros

Cada cérebro se recusa a processar dados fora do seu domínio → **anti-alucinação estrutural**.

## Questões Abertas

- [ ] O protocolo P2P do Crompressor pode servir como transporte para sinapse?
- [ ] Como escalar a comunicação inter-branch para centenas de simulações?
- [ ] Os Cérebros de Domínio podem ser treinados via Codebook Learning?
