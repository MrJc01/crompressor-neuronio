# Quantização em Redes Neurais — Estado da Arte

*Pesquisa realizada em 23/04/2026*

---

## A Pergunta Central do CromGPT

**Pergunta:** É possível treinar uma rede neural onde os pesos não são Float32, mas **índices de um codebook aprendido**?

**Resposta curta:** Sim, mas com ressalvas. O estado da arte mostra 3 abordagens viáveis e 2 que falharam.

---

## 1. VQ-VAE (Vector Quantized Variational Autoencoder)

**Paper:** van den Oord et al. (2017) — DeepMind
**O que faz:** Quantiza o **espaço latente** de um autoencoder usando um codebook treinável.

### Mecanismo
```
Encoder → z_contínuo → argmin(||z - e_k||) → z_quantizado = e_k → Decoder
```

O codebook `e` (shape [K, D]) é treinado junto com o encoder/decoder.

### 3 Losses do VQ-VAE
1. **Reconstruction Loss**: Decoder reconstrói input a partir de `z_quantizado`
2. **Codebook Loss**: Puxa centróides para perto dos outputs do encoder: `||sg[z] - e||²`
3. **Commitment Loss**: Força encoder a "se comprometer" com centróides: `||z - sg[e]||²`

### Relevância para CromLinear
O VQ-VAE provou que codebooks treináveis convergem em redes neurais. A diferença:
- VQ-VAE: codebook no **espaço latente** (entre encoder e decoder)
- CromLinear: codebook nos **pesos** da rede (substitui nn.Linear)

> ⚠️ **Risco:** No VQ-VAE, o codebook quantiza ATIVAÇÕES. No CromLinear, quantizamos PESOS. Pesos têm distribuição diferente.

---

## 2. Straight-Through Estimator (STE)

**Paper:** Bengio et al. (2013) — "Estimating or Propagating Gradients Through Stochastic Neurons"
**O que faz:** Permite backpropagation através de operações não-diferenciáveis (como quantização).

### Mecanismo
```python
# Forward: quantização real
z_q = quantize(z)  # não-diferenciável

# Backward: STE ignora a quantização
# Gradiente flui como se quantize() fosse a identidade
dL/dz ≈ dL/dz_q  # cópia direta
```

### Implementação em PyTorch
```python
# O truque clássico:
z_q = codebook[indices]
z_q = z + (z_q - z).detach()  # Forward: z_q, Backward: z
```

### Relevância para CromLinear
STE é a PRIMEIRA opção para o backward pass. É simples e funciona na prática.

> ⚠️ **Risco:** O gradiente STE é uma APROXIMAÇÃO. Para K grande (muitos centróides), o erro de aproximação pode se acumular.

---

## 3. Gumbel-Softmax (Alternativa ao STE)

**Paper:** Jang et al. (2017) — "Categorical Reparameterization with Gumbel-Softmax"
**O que faz:** Torna a seleção de categorias discretas (ex: qual centróide usar) diferenciável.

### Mecanismo
Em vez de `argmin` (hard), usa softmax com temperatura:
```python
# Soft selection: diferenciável
weights = softmax(-distances / temperature)
z_q = sum(weights * codebook)  # média ponderada
```

Com temperature → 0, converge para one-hot (hard selection).

### Relevância para CromLinear
Gumbel-Softmax é o FALLBACK se STE não funcionar. Mais preciso que STE, mas mais lento (requer computar distâncias para TODOS os K centróides a cada forward pass).

> ⚠️ **Custo:** Para K=256, é 256x mais caro que STE por forward pass. Para K=2048, inviável.

---

## 4. Quantization-Aware Training (QAT)

**Paper:** Jacob et al. (2018) — Google — "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"
**O que faz:** Simula quantização INT8 durante o treinamento para que o modelo aprenda a ser robusto à perda de precisão.

### Mecanismo
```
Forward: W_q = round(W / scale) * scale  # simula INT8
Backward: STE (gradiente ignora o round)
```

### Relevância para CromLinear
QAT prova que treinar com pesos "degradados" funciona — o modelo adapta. CromLinear vai além: em vez de INT8 (256 valores escalares), usamos K vetores D-dimensionais.

> ✅ **Boas notícias:** Se QAT funciona com 256 níveis escalares, CromLinear com K=256 centróides vetoriais deveria funcionar também.

---

## 5. Finite Scalar Quantization (FSQ) — A Alternativa Elegante

**Paper:** Mentzer et al. (2023) — Google — "Finite Scalar Quantization: Simpler Codebook Learning"
**O que faz:** Substitui codebooks aprendidos por uma **grid fixa** de quantização.

### Diferença Fundamental
```
VQ-VAE:  codebook aprendido, K vetores livres → RISCO de codebook collapse
FSQ:     grid fixa, sem aprendizado do codebook → ZERO codebook collapse
```

### Comparação

| Feature | VQ (Learned) | FSQ (Fixed Grid) |
|---------|-------------|-------------------|
| Codebook | Aprendido | Fixo |
| Collapse | Comum | **Zero** |
| Auxiliary losses | Sim (commitment) | **Não** |
| Utilização | Baixa (~30-60%) | **~100%** |
| Flexibilidade | Alta | Média |

### Relevância para CromLinear
FSQ sugere uma VARIANTE da CromLinear: em vez de aprender os centróides, usar uma grid fixa e aprender apenas ONDE cada peso cai na grid.

> 💡 **Ideia para explorar:** `CromLinearFSQ` — versão com grid fixa. Zero risco de collapse. Mais estável no treinamento.

---

## 6. Binary/Ternary Neural Networks

**Paper:** Courbariaux et al. (2016) — "BinaryConnect"
**O que faz:** Pesos são literalmente -1 ou +1 (binário) ou -1, 0, +1 (ternário).

### Resultados
- MNIST: ~99% (quase igual a Float32)
- ImageNet: ~50-60% top-1 (muito abaixo de Float32 ~76%)
- Conclusão: funciona para tarefas simples, perde muito em tarefas complexas

### Relevância para CromLinear
Mostra o PIOR CASO de quantização extrema. CromLinear com K=256 é MUITO mais expressivo que binário (2 valores). Se binário atinge 99% em MNIST, CromLinear deveria atingir >95%.

---

## Decisão para CromLinear

```
┌───────────────────────────────────────────────────────────────────┐
│  PLANO DE IMPLEMENTAÇÃO:                                          │
│                                                                   │
│  1. Implementar CromLinear com STE (opção padrão)                │
│  2. Adicionar commitment loss + codebook loss (anti-collapse)    │
│  3. Se collapse: experimentar FSQ (grid fixa)                    │
│  4. Se gradientes ruidosos: experimentar Gumbel-Softmax          │
│  5. Inicializar codebook com K-Means++ nos pesos Float32         │
│                                                                   │
│  Gradiente: STE → Gumbel-Softmax (fallback)                     │
│  Codebook: Aprendido → FSQ fixed grid (fallback)                 │
└───────────────────────────────────────────────────────────────────┘
```
