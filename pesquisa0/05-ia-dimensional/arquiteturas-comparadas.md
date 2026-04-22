# Arquiteturas Comparadas: 4D vs 5D

## Tabela Comparativa Principal

| Recurso | 4D (LLM Atual / Transformer) | 5D (Agente Preditivo / World Model) |
|:--------|:------------------------------|:------------------------------------|
| **Processamento** | Sequencial/Bloco (Context Window) | Ramificado (Tree Search / MCTS) |
| **Lógica** | Probabilidade de Token | Inferência de Causalidade |
| **Paper Base** | Attention Is All You Need (2017) | Tree of Thoughts (2023) / JEPA (2024) |
| **Estado** | Reativo ao Input | Proativo/Simulativo (Active Inference) |
| **Soberania** | Dependente de Dados Históricos | Dependente de Simulação Interna |
| **Raciocínio** | Sistema 1 (rápido, associativo) | Sistema 2 (deliberativo, analítico) |
| **Erros** | Detectados após gerar | Prevenidos antes de gerar |
| **Backtracking** | ❌ Impossível | ✅ Nativo |
| **Memória** | KV Cache linear O(n) | KV Cache otimizado O(1) via Crompressor |
| **Analogia Git** | `git log --oneline` | `git log --all --graph` |
| **Analogia Física** | Universo de Bloco (4D) | Multiverso Computacional (5D) |

## Espectro Dimensional das Arquiteturas

```
1D ────── 2D ────── 3D ────── 4D ────── 5D
RNN       CNN       GNN       Transformer  ToT/MCTS
│         │         │         │            │
Sequência Grades    Grafos    Atenção      Árvore de
temporal  espaciais           plena        possibilidades
```

## O Papel do Crompressor na Transição

| Gargalo da 4D | Solução Crompressor para 5D |
|:--------------|:--------------------------|
| KV Cache cresce O(n) com contexto | Codebook comprime para O(1) |
| Branches multiplicam memória | Delta storage: 99.99% compartilhado |
| Pruning desperdiça computação | Merkle Tree: localização instantânea |
| Alucinações contaminam output | Security: sandboxing de branches |

## Questões Abertas

- [ ] Existe uma "6D computacional"? O que seria?
- [ ] A transição 4D→5D requer novo hardware ou apenas novo software?
- [ ] O Crompressor é condição necessária ou apenas otimização para 5D?
