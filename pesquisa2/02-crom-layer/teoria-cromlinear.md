# Teoria da CromLinear — Matemática Completa

*O coração da Pesquisa 2. Se esta camada funciona, tudo funciona.*

---

## 1. O Problema

Uma camada `nn.Linear` faz:
```
y = x · W + b
```
Onde `W` é uma matriz Float32 de shape `[in_features, out_features]`.

**Problema:** `W` ocupa `in × out × 4 bytes`. Para dim=768:
- Uma camada: 768 × 768 × 4 = **2.36 MB**
- 12 layers × 6 matrizes = 72 matrizes = **170 MB** só de pesos lineares

**Proposta:** E se `W` não fosse uma matriz Float32, mas sim um **codebook + índices**?

---

## 2. A Solução: CromLinear

### Definição

```python
class CromLinear(nn.Module):
    def __init__(self, in_features, out_features, K=256, D=64):
        """
        K: número de centróides no codebook
        D: dimensão de cada centróide
        """
        # Codebook treinável: K vetores de D dimensões
        self.codebook = nn.Parameter(torch.randn(K, D))  # shape [K, D]
        
        # Número de blocos que a matriz W é dividida
        n_elements = in_features * out_features
        n_blocks = n_elements // D
        
        # Índices: qual centróide cada bloco usa (não treinável diretamente)
        self.indices = nn.Parameter(torch.randint(0, K, (n_blocks,)), requires_grad=False)
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_features))
```

### Forward Pass

```python
def forward(self, x):
    # 1. Reconstruir W a partir do codebook
    W_flat = self.codebook[self.indices]  # shape [n_blocks, D]
    W = W_flat.reshape(self.in_features, self.out_features)
    
    # 2. Multiplicação normal
    return x @ W + self.bias
```

### Backward Pass (Straight-Through Estimator)

O problema: `self.codebook[self.indices]` é um lookup discreto. O gradiente de `argmin` é zero.

Solução STE:
```python
# Em vez de:
W_flat = self.codebook[self.indices]

# Fazemos:
W_continuous = compute_continuous_assignment(x)  # soft version
W_quantized = self.codebook[self.indices]
W_flat = W_continuous + (W_quantized - W_continuous).detach()
# Forward usa W_quantized, Backward usa W_continuous
```

---

## 3. Compressão: Por que funciona?

### Comparação de Memória

Para uma camada 768→768:

| Representação | Fórmula | Bytes |
|--------------|---------|-------|
| Float32 | 768 × 768 × 4 | **2,359,296** |
| CromLinear K=256, D=64 | (256 × 64 × 4) + (9216 × 1) | **74,944** |
| **Compressão** | | **31.5x menor** |

O codebook (256 centróides × 64 dims × 4 bytes = 65KB) é compartilhado, e cada bloco de 64 valores é representado por 1 byte (índice 0-255).

### Tabela de Compressão por K e D

| K | D | Codebook (KB) | Índices (KB) | Total (KB) | Compressão |
|---|---|--------------|-------------|-----------|------------|
| 64 | 32 | 8 | 72 | 80 | 29x |
| 128 | 64 | 32 | 36 | 68 | 34x |
| 256 | 64 | 64 | 36 | 100 | 23x |
| 512 | 64 | 128 | 72 | 200 | 11x |
| 1024 | 64 | 256 | 72 | 328 | 7x |

> **Ponto ideal:** K=256, D=64 → 23-31x de compressão com expressividade razoável.

---

## 4. Treinamento: Como o codebook aprende?

### Opção A: STE Puro
```
1. Forward: lookup codebook[indices] → W → y = x·W
2. Loss: CrossEntropy(y, target)
3. Backward: gradiente flui até codebook via STE
4. Update: optimizer.step() atualiza codebook
5. Re-assign: periodicamente, recalcular indices via nearest neighbor
```

### Opção B: End-to-End com EMA
```
1. Forward: codebook[indices] → W → y
2. Codebook update: EMA dos vetores mais próximos (como VQ-VAE)
   e_new = γ * e_old + (1-γ) * mean(z_assigned)
3. Commitment loss: ||z - sg[e]||² (força estabilidade)
```

### Opção C: Gumbel-Softmax (Fallback)
```
1. Forward: soft_assign = softmax(-||z - codebook|| / τ)
           W = Σ(soft_assign * codebook)  # média ponderada
2. Backward: gradientes naturais (diferenciável!)
3. Temperature annealing: τ: 1.0 → 0.1 ao longo do treino
```

---

## 5. Riscos e Mitigações

| Risco | Probabilidade | Mitigação |
|-------|--------------|-----------|
| Codebook collapse | Alta | Commitment loss + EMA update + monitorar utilização |
| Gradientes ruidosos (STE) | Média | Gradient clipping + LR menor para codebook |
| Não convergência | Média | Começar com K pequeno (64), aumentar gradualmente |
| Lentidão no forward | Baixa | Lookup é O(1), mais rápido que matmul para K pequeno |

---

## 6. Diagrama de Fluxo

```
Input x [batch, in_features]
    │
    ▼
┌─────────────────────────┐
│  CromLinear              │
│                         │
│  codebook [K, D]  ──┐  │
│  indices [n_blocks] ─┤  │
│                      │  │
│  W = codebook[idx]   │  │
│  W = reshape(in,out) │  │
│                      │  │
│  y = x @ W + bias    │  │
└──────────┬──────────┘
           │
           ▼
    Output y [batch, out_features]
```

---

## 7. Diferenças vs Estado da Arte

| Técnica | O que quantiza | Quando | Nosso diferencial |
|---------|---------------|--------|-------------------|
| VQ-VAE | Espaço latente | Treino | CromLinear quantiza **pesos** |
| QAT | Pesos (INT8) | Treino | CromLinear usa **codebook vetorial** (mais expressivo) |
| GPTQ | Pesos (INT4) | Pós-treino | CromLinear é **nativo** (pesos nascem quantizados) |
| AWQ | Pesos (INT4) | Pós-treino | Idem acima |
| **CromLinear** | **Pesos (codebook)** | **Treino** | **Combina VQ + QAT: codebook vetorial nativo** |

> Se funcionar, é genuinamente novo: ninguém treinou um LLM onde os pesos são codebooks vetoriais desde o início.
