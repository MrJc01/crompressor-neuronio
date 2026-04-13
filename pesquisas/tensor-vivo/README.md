# 🔬 Tensor-Vivo: Crompressor Como Substituto de Tensores

> **Linha de Pesquisa Primordial:** Investigar se a representação comprimida do Crompressor (CDC + Codebook DNA + Merkle) pode **substituir diretamente** os pesos numéricos dos tensores em um modelo de IA, em vez de apenas compactá-los como arquivo.

---

## A Pergunta Central

A abordagem convencional de IA é:

```
Tensor = array de float32 → [0.0231, -0.1492, 0.8831, ...]
```

A pergunta desta pesquisa é:

> **E se, em vez de números, o "peso" de cada neurônio fosse o próprio sentido
> comprimido pelo Crompressor?**

Em vez de `float32[4096]`, cada neurônio teria um **chunk CDC** que carrega a
semântica comprimida em DNA Base-4 com Codebook treinável. A "ativação" do
neurônio não seria uma multiplicação matricial, mas uma **descompressão
seletiva + lookup no codebook**.

## O Que Isso Muda

| Aspecto | Tensor Clássico | Tensor-Vivo (Crompressor) |
|---|---|---|
| Representação | `float32[N]` | `Chunk CDC (DNA + Codebook ref)` |
| Operação | dot product | decompress + semantc lookup |
| Treinamento | backprop sobre floats | ajuste de Codebook entries |
| Peso | valor numérico | sentido comprimido |
| Tamanho | N × 4 bytes | chunk dedupado (variável) |
| Interpretabilidade | Opaco | Potencialmente legível via DNA decode |

## Hipóteses a Investigar

### H1: Representação
Pode um chunk CDC de ~512 bytes (DNA Base-4) capturar a mesma informação
que um vetor de embedding de 896 floats (3584 bytes)?

### H2: Forward Pass
O que acontece se substituirmos a multiplicação matricial `W × x` por
`decompress(chunk[i]) · x`? O resultado converge para algo interpretável?

### H3: Treinamento
Se modificarmos apenas as entradas do Codebook (sem tocar nos chunks), o
modelo "aprende"? Isso seria equivalente a LoRA, mas no espaço do Codebook.

### H4: Composição
Dois chunks CDC podem ser "compostos" (XOR/merge) para gerar um chunk que
represente a fusão semântica de dois conceitos diferentes?

## Experimentos Planejados

### Exp 1: Embedding Substitutivo (micro-escala)
```
1. Treinar um MLP simples (MNIST ou similar)
2. Extrair a camada de embedding (ex: 128 × 64 floats)
3. Comprimir cada row de 64 floats com FastCDC → chunk
4. No forward pass, substituir o lookup de embedding por:
   embedding[token] = decode_dna(codebook[chunk_hash[token]])
5. Medir accuracy: original vs crompressor-substituted
```

### Exp 2: Peso-como-Sentido
```
1. Pegar os pesos de uma camada linear (ex: 256 × 128 floats)
2. Para cada neurônio (row de 128 floats), gerar um chunk CDC
3. Testar: chunk_similarity(neuronio_A, neuronio_B) correlaciona
   com cosine_similarity(peso_A, peso_B)?
4. Se sim: a estrutura semântica do codebook preserva a geometria do espaço de pesos
```

### Exp 3: Codebook-as-LoRA
```
1. Congelar todos os chunks de um modelo comprimido
2. Permitir apenas modificação das entries do Codebook
3. Fine-tune via gradient descent no espaço do Codebook
4. Medir: o modelo adapta? Qual a expressividade do Codebook?
```

## Stack Técnico Provável

- **Go**: Processamento CDC, Codebook, DNA encode/decode
- **Python (PyTorch)**: Treinar modelos pequenos, medir accuracy
- **Interface Go↔Python**: via arquivo intermediário ou gRPC

## Status

🔴 **Não iniciado** — Esta é a ideia primordial do projeto. A pesquisa
Cérebro-FUSE veio da vertente "modelo inteiro como neurônio", mas esta
vertente investiga algo mais fundamental: **o tensor em si pode ser substituído**.

---

*"O neurônio que comprime é o neurônio que pensa — mas e se o próprio peso for compressão?"*
