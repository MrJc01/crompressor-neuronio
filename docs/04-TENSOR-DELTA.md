# ⚡ Tensor Delta: XOR + Vector Quantization sobre .crom

> *"O delta é o pensamento. O cérebro é a memória."*

---

## Conceito Central

O **Tensor Delta** é o mecanismo pelo qual um neurônio fixo (brain.crom) gera saídas **não-determinísticas**. Em vez de retreinar o modelo, aplicamos uma transformação diferencial (delta) sobre os chunks comprimidos do codebook.

```
Saída = F(Cérebro_fixo ⊕ Tensor_Delta)

onde ⊕ pode ser:
  - XOR bitwise (Vertente 1: rápido, determinístico)
  - Adição vetorial (Vertente 2: granular, adaptativo)
  - Combinação ponderada (Vertente 3: multi-brain)
```

---

## Operação 1: XOR Delta

### Fundamento
O XOR Delta já existe no core do Crompressor e no Crompressor-Sinapse (Frente 3). A operação é simples:

```go
// Pseudocódigo
func ApplyXORDelta(chunk []byte, delta []byte) []byte {
    result := make([]byte, len(chunk))
    for i := range chunk {
        result[i] = chunk[i] ^ delta[i%len(delta)]
    }
    return result
}
```

### Propriedades Matemáticas
- **Reversível:** `A ⊕ B ⊕ B = A` → sempre pode reconstruir original
- **Comutativa:** `A ⊕ B = B ⊕ A`
- **Associativa:** `(A ⊕ B) ⊕ C = A ⊕ (B ⊕ C)`
- **Identidade:** `A ⊕ 0 = A` → delta zero = neurônio original
- **O(n):** Operação linear, sem overhead computacional

### Alinhamento com Paper ZipLLM BitX (2025)
O ZipLLM demonstrou que modelos fine-tuned da mesma família compartilham **diferenças XOR altamente esparsas**. Exatamente o que nosso delta faz:

```
Delta entre base e fine-tuned ≈ 3-8% dos bytes são não-zero
→ O delta comprimido é minúsculo comparado ao modelo
→ BitX (do ZipLLM) codifica exatamente isso
```

---

## Operação 2: Vector Quantization (VQ)

### Fundamento
Em vez de XOR binário, quantizamos os vetores do codebook em um espaço discreto e aplicamos deltas nesse espaço quantizado, inspirados no **LLVQ (Leech Lattice VQ, Mar 2026)**.

```go
type VectorQuantizer struct {
    Codebook    [][]float32  // centroides do codebook
    Dimension   int          // dim de cada vetor
    NumClusters int          // número de centroides
}

// Delta no espaço quantizado
func (vq *VectorQuantizer) ApplyDelta(chunkID int, delta []float32) []float32 {
    original := vq.Codebook[chunkID]
    result := make([]float32, vq.Dimension)
    for i := range result {
        result[i] = original[i] + delta[i]  // adição vetorial
    }
    return result
}
```

### Vantagem sobre XOR
- **Granularidade semântica:** delta pode ser parcial (afeta apenas certas dimensões)
- **Interpolação:** delta pode ser fracionário (0.3 do delta original)
- **Composição:** múltiplos deltas podem ser somados linearmente

### Alinhamento com TurboQuant (ICLR 2026)
TurboQuant aplica VQ online no KV cache. Nosso VQ Neural faz o mesmo sobre o Codebook inteiro do .crom:

```
TurboQuant: KV_cache → VQ → 6x redução
Neurônio:   Codebook → VQ → delta no espaço comprimido
```

---

## Operação 3: Multi-Tensor Composition

### Fundamento
Para o Multi-Brain Routing, múltiplos deltas de neurônios diferentes são compostos via ponderação:

```go
type ComposedDelta struct {
    Deltas  [][]byte    // deltas de cada neurônio
    Weights []float32   // pesos de routing
}

func (cd *ComposedDelta) Compose() []byte {
    result := make([]byte, len(cd.Deltas[0]))
    for i := range result {
        var weighted float32
        for j, delta := range cd.Deltas {
            weighted += float32(delta[i]) * cd.Weights[j]
        }
        result[i] = byte(weighted)
    }
    return result
}
```

### Alinhamento com Brainstacks (2026)
O Brainstacks usa routing sigmoid para compor adapter stacks. Nosso Multi-Brain faz o mesmo com neurônios .crom:

```
Brainstacks: Σ(weight_i × adapter_stack_i) → output
Neurônio:    Σ(weight_i × brain_i.crom XOR delta_i) → output
```

---

## Formato do Tensor Delta (Proposto)

```
┌────────────────────────────────────────┐
│ DELTA HEADER (32 bytes)                │
│ ┌──────────────────────────────────┐   │
│ │ Magic: "DELT" (4 bytes)          │   │
│ │ Version: 0x01 (1 byte)           │   │
│ │ Type: XOR|VQ|COMPOSED (1 byte)   │   │
│ │ TargetBrainHash: [16]byte        │   │
│ │ DeltaSize: uint32 (4 bytes)      │   │
│ │ NonZeroRatio: float32 (4 bytes)  │   │
│ │ Reserved: (2 bytes)              │   │
│ └──────────────────────────────────┘   │
│                                        │
│ DELTA DATA (variável)                  │
│ ┌──────────────────────────────────┐   │
│ │ Para XOR: sparse bitmap + values │   │
│ │ Para VQ: centroid offsets         │   │
│ │ Para COMPOSED: weights + refs    │   │
│ └──────────────────────────────────┘   │
│                                        │
│ SIGNATURE (opcional, 64 bytes)         │
│ ┌──────────────────────────────────┐   │
│ │ Dilithium signature do delta     │   │
│ └──────────────────────────────────┘   │
└────────────────────────────────────────┘
```

---

## Hipóteses a Testar

| # | Hipótese | Como Medir |
|:---|:---|:---|
| H1 | Delta XOR < 5% do tamanho do cérebro | `len(delta) / len(brain.crom)` |
| H2 | VQ Delta preserva qualidade > 90% | BLEU score vs. modelo original |
| H3 | Composição multi-delta é linear | `compose(A,B) ≈ compose(B,A)` |
| H4 | Entropia do delta < entropia do cérebro | Shannon entropy de ambos |
| H5 | XOR é 10x mais rápido que VQ | Benchmark ns/op |
| H6 | Delta esparsificação > 80% dos bytes são zero | Count non-zero bytes |

---

> **Próximo:** [05 — Integração Ecossistema](05-INTEGRACAO-ECOSSISTEMA.md)
