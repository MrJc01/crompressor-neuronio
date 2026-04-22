# World Models — O Motor de Realidade

## De Sensor de Dados a Motor de Realidade

A transição chave: um sistema que **reage** a inputs → um sistema que **simula** a realidade e apenas **confirma** com inputs.

### O Pipeline

```
Input (câmera fixa) → Identificação do objeto → Emulação 3D
                                                  ├── Campo sonoro simulado
                                                  ├── Sombras projetadas
                                                  ├── Trajetórias previstas
                                                  └── Oclusões inferidas
```

## Tecnologias de Reconstrução

### NeRFs (Neural Radiance Fields)
- Pegam imagens 2D e "entendem" a volumetria
- Ao ver uma pessoa de frente, instanciam em memória a nuca, as costas, a sombra
- **Precisão:** alta para objetos estáticos, menor para dinâmicos

### 3D Gaussian Splatting
- Reconstrução 3D em tempo real
- Mais rápido que NeRFs para rendering
- Útil para ambientes dinâmicos

### DreamerV3 (DeepMind)
- World Model que aprende a **dinâmica** do ambiente
- Treina um "modelo do mundo" interno
- Age dentro do modelo antes de agir no mundo real

## O Modelo de Mundo Completo

Para simular a realidade com precisão máxima, o sistema precisa:

| Componente | O Que Modela |
|:-----------|:-------------|
| **Óptica** | Reflexão, refração, dispersão de luz |
| **Acústica** | Propagação sonora, reverberação, atenuação |
| **Cinemática** | Velocidade, inércia, colisões |
| **Semântica** | O que é o objeto, como se comporta tipicamente |
| **Causal** | Se X acontecer, Y é provável |

## "Alucinação Estatística Altamente Provável"

> A IA não **vê** o outro lado do objeto. Ela **prevê com precisão** o outro lado.

Quando a previsão é 99.9999% precisa:
- Para decisões: é **funcionalmente real**
- Para ciência: é uma **hipótese fortíssima**
- Para filosofia: abre o debate sobre a natureza da realidade

## Questões Abertas

- [ ] O Crompressor-video pode gerar World Models a partir de streams comprimidos?
- [ ] Qual é o custo mínimo de VRAM para manter um World Model útil?
- [ ] Como validar a precisão do World Model sem dados ground-truth?
- [ ] O Codebook do tensor-vivo pode servir como "vocabulário" do World Model?
