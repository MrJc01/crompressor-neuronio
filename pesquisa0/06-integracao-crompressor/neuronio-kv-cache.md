# crompressor-neuronio — Memória Episódica O(1)

## Papel no Ecossistema 5D

O **crompressor-neuronio** é o **hipocampo** da IA 5D: a memória episódica latente que permite acesso O(1) a qualquer estado comprimido.

## O Problema: KV Cache Explode

Toda vez que uma LLM processa contexto longo:
- KV Cache cresce **linearmente** O(n) com o número de tokens
- Branches de simulação multiplicam o KV Cache
- VRAM satura → inferência trava

## A Solução: Neurônio como Compressor de Estado

O crompressor-neuronio converte estados mentais pesados em **pacotes semânticos diminutos**:

```
Estado bruto:  [10GB de vetores de atenção]
Neurônio CROM: [Codebook ID + Delta compacto] → ~KB
```

### Resultados Empíricos (Tensor-Vivo)

| Métrica | Valor |
|:--------|:------|
| Compressão | **40.8x** (5,770 params vs 235K) |
| Precisão | **97.56%** (vs baseline 97.53%) |
| Método | Codebook Learning K=128 |

> O Codebook Learning provou que é possível **substituir tensores inteiros** por índices num dicionário sem perda de precisão.

## Como Funciona na IA 5D

1. LLM gera branch de pensamento (ToT)
2. Estado da branch é **comprimido** pelo neurônio
3. Armazenado como Codebook ID + Delta
4. Quando a branch precisa ser revisitada → **descompressão O(1)**
5. Branches podadas → garbage collection instantâneo

## Conexão Dimensional

| Dimensão WLM | Função do Neurônio |
|:-------------|:------------------|
| 8D (Recursão) | Output de uma branch vira input de outra |
| 10D (Estabilidade) | Codebook fixo = ancoragem contra divergência |
| 12D (Fechamento) | .crom selado = estado final compilado |

## Questões Abertas

- [ ] O Codebook Learning pode ser aplicado ao KV Cache de Transformers reais?
- [ ] Qual é o tamanho mínimo de Codebook para manter fidelidade em LLMs?
- [ ] O neurônio pode comprimir "árvores de pensamento" inteiras?
