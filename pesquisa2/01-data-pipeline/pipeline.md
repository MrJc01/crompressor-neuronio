# Pipeline de Dados — Fluxo Completo

```
Download → Filtro Idioma → Filtro Qualidade → Deduplicação → Tokenização → Formato Final
```

---

## Etapa 1: Download
```python
from datasets import load_dataset
wiki = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train")
carolina = load_dataset("carolina-c4ai/corpus-carolina", split="train")
```

## Etapa 2: Filtro de Idioma
- Usar `fasttext lid.176.bin` para detectar PT
- Remover textos que não sejam >95% PT

## Etapa 3: Filtro de Qualidade
- Comprimento mínimo: >50 caracteres
- Remover boilerplate web (menus, footers, cookies)
- Remover URLs e código HTML residual
- Filtrar por perplexidade (opcional, se tempo permitir)

## Etapa 4: Deduplicação
- Exact dedup: hash SHA-256 de cada documento
- Fuzzy dedup: MinHash com Jaccard >0.8

## Etapa 5: Tokenização
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("pierreguillou/gpt2-small-portuguese")
```

## Etapa 6: Formato Final
- Salvar como arquivo tokenizado (numpy .npy ou PyTorch .pt)
- Cada sequência: 512 tokens
- Concatenar documentos com token `<|endoftext|>` como separador

## Estatísticas Esperadas
- Total tokens: ~400-500M
- Vocab size: ~50K (tokenizador existente)
- Número de sequências: ~800K-1M (de 512 tokens cada)
