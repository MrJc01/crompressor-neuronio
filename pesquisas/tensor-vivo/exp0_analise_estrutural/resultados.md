# Exp0 — Resultados: Análise Estrutural CDC

## Dados

| Análise | Resultado |
|---|---|
| Modelo | MLP 784→256→128→10 (235K params, 940KB) |
| Accuracy Baseline | **97.53%** |
| CDC params | min=64, avg=512, max=4096 |
| Total chunks (global) | 1750 |
| Chunks únicos | 1750 |
| **Taxa de Dedup CDC** | **0.00%** |
| Entropia média | 6.81 bits/byte |
| Cross-layer dedup | 0 hashes compartilhados |

## Análise Refinada (3 estratégias)

| Estratégia | Resultado |
|---|---|
| A: CDC chunks menores (avg=256) | 0% dedup em todas as camadas |
| B: Quantizar 8-bit antes do CDC | 0% dedup (entropia caiu para 5.84) |
| B: Quantizar 4-bit antes do CDC | 0% dedup (entropia caiu para 3.44) |
| C: Hash por neurônio (row) | 0 duplicados exatos |
| C: Cosine > 0.95 entre neurônios | **0 pares** em todas as camadas |
| C: Cosine > 0.99 entre neurônios | 0 pares |

## Conclusão

**CDC hash exato NÃO encontra dedup em pesos de redes neurais treinadas.**

Cada neurônio é numericamente único — mesmo com quantização agressiva (4-bit),
os padrões de bytes nunca colidem no nível de hash.

**PORÉM**, a entropia de 6.81 bits/byte (vs 8.0 máximo teórico) confirma que
os pesos **não são ruído aleatório** — existe estrutura comprimível.

**Insight crítico:** A dedup do Crompressor sobre tensores não deve ser por
hash exato, mas por **proximidade no espaço de pesos** (clustering).
Isso é validado no Exp1.
