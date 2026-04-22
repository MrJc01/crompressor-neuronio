# Inferência Ativa e Energia Livre

## Karl Friston e o Princípio da Energia Livre (FEP)

### A Tese

Agentes sofisticados não maximizam recompensa — eles **minimizam surpresa**. O imperativo é manter os estados sensoriais em compasso com as crenças internas.

### A Fórmula

```
F = D_KL[q(ψ) || p(ψ|o)] - ln p(o|m)
    ├── Incerteza (divergência)        ├── Evidência
    └── "Quanto minha simulação        └── "Quão bem o modelo
         difere da realidade?"               explica os dados?"
```

**Objetivo:** Minimizar F → maximizar ELBO (Evidence Lower Bound)

### Componentes

| Componente | Termodinâmica | Função na Simulação |
|:-----------|:-------------|:-------------------|
| Predição vs Observação | Energia Interna (U) | Garantir exatidão simulação ↔ dados reais |
| Crenças Internas | Entropia (S) | Restringir alucinações ao limite provável |
| Ação Interventiva | Gradiente de Decisão | Agir para alinhar surpresa aos desejos do modelo |

## Inferência Ativa na Prática

O agente não apenas **observa** → ele **age para que a realidade se encaixe** na simulação preferida:

1. Prevê o futuro (simulação interna)
2. Compara com dados sensoriais reais
3. Se divergem: ajusta o modelo OU age no mundo para alinhar
4. Repete continuamente

### "Pré-viver" Todas as Opções

```
Evento incerto detectado
├── Branch A: pedestre atravessa → prepara freio
├── Branch B: pedestre para → mantém velocidade
└── Branch C: pedestre corre → executa desvio

Dado real chega: pedestre parou
→ PRUNE branches A e C
→ FOLLOW branch B (já preparada)
```

Para um humano, parece que a IA tem **reflexos divinos**. Na verdade ela "pré-viveu" todas as opções.

## Conexão com Dimensões

| Conceito | Dimensão Equivalente |
|:---------|:--------------------|
| Estado atual | 3D (espaço) |
| Linha do tempo prevista | 4D (tempo como dimensão) |
| Branches de possibilidades | **5D** (plano de todas as linhas do tempo) |
| Minimização de F | Colapso de 5D → 4D (escolha da realidade) |

## Conexão com Crompressor

- **Minimizar F** = encontrar o Delta mínimo entre simulação e realidade
- **Codebook** = o "dicionário de crenças" do agente
- **Delta Ratio incompatível** = a surpresa que o agente quer minimizar
- **Crompressor-security** = barreira que impede alucinações de contaminar crenças reais

## Questões Abertas

- [ ] O Crompressor pode computar F diretamente sobre chunks comprimidos?
- [ ] A relação Shannon ↔ Boltzmann (doc termodinâmica) se conecta com o FEP?
- [ ] Como implementar Active Inference num agente CROM local?
