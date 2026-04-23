# CromGPT — Guia Colab

## Instruções Rápidas

### 1. Abrir no Colab
- Vá em [colab.google.com](https://colab.google.com)
- **File → Upload notebook** ou crie um novo
- **Runtime → Change runtime type → T4 GPU**

### 2. Célula 1: Setup
```python
!git clone https://github.com/MrJc01/crompressor-neuronio.git
%cd crompressor-neuronio/pesquisa2
!pip install -q transformers datasets
```

### 3. Célula 2: Executar script completo
```python
%run colab/cromgpt_full_train.py
```

### OU executar por partes (recomendado):

Copie cada seção do `cromgpt_full_train.py` em células separadas:

| Célula | Conteúdo | Tempo estimado |
|--------|----------|---------------|
| 1 | Setup + imports | ~1 min |
| 2 | Download Wikipedia PT | ~15-30 min |
| 3 | Importar modelos | ~1 min |
| 4 | Dataset + DataLoader | ~1 min |
| 5 | **Treino (CromGPT + Baseline)** | **~2-6h** |
| 6 | Avaliação completa | ~5 min |
| 7 | Salvar resultados | ~1 min |

## Parâmetros para Ajustar

Se der **OOM (Out of Memory)**:
```python
BATCH_SIZE = 4       # reduzir de 8 para 4
SEQ_LEN = 128        # reduzir de 256 para 128
MAX_ARTICLES = 50000 # reduzir dataset
```

Se quiser treino **mais rápido** (menos preciso):
```python
epochs = 1           # em vez de 3
MAX_ARTICLES = 30000 # menos dados
```

## O que o script faz

1. **Baixa** ~100K artigos da Wikipedia PT (~50-100M tokens)
2. **Tokeniza** com `pierreguillou/gpt2-small-portuguese`
3. **Treina** CromGPT (12 layers, d=768, K=256, D=64) com FP16
4. **Treina** Baseline idêntico com nn.Linear puro
5. **Avalia**: PPL, velocidade, VRAM, tamanho disco, 10 prompts PT
6. **Salva** modelo .crom v3 + resultados JSON

## Itens do PLANEJAMENTO Cobertos

- [x] 3.2.8 — Mixed precision (FP16)
- [x] 3.3.1-5 — Treinamento efetivo
- [x] 4.2.1 — Perplexidade
- [x] 4.2.2 — Diversidade lexical
- [x] 4.2.3 — Tamanho em disco
- [x] 4.2.4 — Velocidade de inferência
- [x] 4.2.5 — VRAM
- [x] 4.3.1 — 10 prompts PT
- [x] 4.3.4 — Comparação lado-a-lado

## Após o Colab

Baixe o `colab_full_results.json` e coloque em `pesquisa2/resultados/`.
Isso alimentará o `papel1.md` com dados reais de GPU.
