# Eixo 05 — Formato .crom v3

> Serialização nativa de modelos completos no formato .crom.

## Objetivo

Definir e implementar o formato `.crom v3` capaz de armazenar um modelo CromGPT inteiro — não apenas codebooks individuais como na v2.

## Evolução do Formato

| Versão | Pesquisa | Conteúdo |
|--------|----------|----------|
| v1 | pesquisa0 | Codebook único + índices (lab06) |
| v2 | pesquisa1 | Header + codebook + índices + MMap (compressor.py) |
| **v3** | **pesquisa2** | **Modelo completo: embeddings + N codebooks + N índices + LM head** |

## Estrutura Proposta

```
.crom v3
├── Header (fixo)
│   ├── Magic: "CROM"
│   ├── Version: 3
│   ├── n_layers, n_heads, dim
│   ├── K, D (codebook params)
│   └── vocab_size
├── Token Embedding (Float16)
├── Position Embedding (Float16)
├── Layers[0..N]
│   ├── Codebook_attn [K, D*4] (Q,K,V,O)
│   ├── Indices_attn [n_blocks]
│   ├── Codebook_ffn [K, D*2] (up, down)
│   ├── Indices_ffn [n_blocks]
│   ├── LN1 weights + bias
│   └── LN2 weights + bias
└── LM Head (Float16 ou codebook)
```

## Lab Associado

`labs/lab29-crom-v3-format/` — save/load + validação round-trip.

## Critério de Conclusão

Salvar modelo → carregar → gerar texto → output idêntico ao original.
