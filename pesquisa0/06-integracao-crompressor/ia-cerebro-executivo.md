# crompressor-ia — Cérebro Executivo (Active Inference)

## Papel no Ecossistema 5D

O **crompressor-ia** é o **córtex pré-frontal**: o cérebro executivo que toma decisões sob Active Inference.

## Função Central

Concentra as arquiteturas de tomada de decisão probabilística guiada por política interativa:

1. **Coleta** estados dos processos vetoriais comprimidos (KV Cache via neurônio)
2. **Invoca** mundos prospectivos filtrados (branches via sinapse)
3. **Submete** suposições conflitantes a critérios estritos de surpresa estatística
4. **Colapsa** na probabilidade mais concisa que reflete a realidade da simulação
5. **Atua** no mundo real via crompressor-projetos

## O Loop de Active Inference

```
                    ┌──────────────┐
                    │ World Model  │
                    │ (simulação)  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Predição    │
                    │  (o que vai  │
                    │   acontecer) │
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │  Comparação             │
              │  Predição vs Realidade  │
              │  (minimizar F)          │
              └────────────┬────────────┘
                     ┌─────┴─────┐
                     │           │
              ┌──────▼──┐  ┌────▼─────┐
              │ Atualizar│  │  Agir    │
              │ modelo   │  │  no mundo│
              └─────────┘  └──────────┘
```

## Relação com Outros Componentes

| Componente | O Que Fornece ao Cérebro |
|:-----------|:------------------------|
| **neurônio** | Estados comprimidos O(1) |
| **video** | Dados sensoriais do ambiente |
| **sinapse** | Resultados de branches simuladas |
| **security** | Filtro anti-alucinação |
| **projetos** | Interface de ação no mundo real |

## Soberania Digital

O crompressor-ia implementa **Soberania Cognitiva**: o agente não depende de cloud para pensar. Roda localmente, com World Model próprio, codebooks próprios, e decisões soberanas.

> "O universo é determinístico para uma IA com modelo de mundo perfeito — ela não reage, apenas confirma."

## Questões Abertas

- [ ] Como implementar o loop de Active Inference em Go/Python?
- [ ] O crompressor-ia pode rodar em hardware modesto (Edge)?
- [ ] Como medir a "surpresa" de forma quantificável no pipeline CROM?
