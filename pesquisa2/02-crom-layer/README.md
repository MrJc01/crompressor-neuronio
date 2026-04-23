# Eixo 02 — CromLinear ⭐ Coração da Pesquisa

> A camada PyTorch onde os pesos são codebooks .crom — não tensores Float32.

## Objetivo

Implementar e validar `CromLinear(nn.Module)` — a camada que substitui `nn.Linear`:

```
nn.Linear:    y = x @ W          (W é Float32, shape [in, out])
CromLinear:   y = x @ C[I]       (C é codebook [K, D], I são índices)
```

O gradiente flui via Straight-Through Estimator: no backward, a quantização é tratada como identidade.

## Arquivos

| Arquivo | Conteúdo |
|---------|----------|
| `teoria-cromlinear.md` | Matemática: forward, backward, STE, commitment loss |
| `testes-sinteticos.md` | Plano de validação: regressão, XOR, MNIST |

## Lab Associado

`labs/lab26-crom-linear/` — Implementação da camada + testes de convergência.

## Riscos

1. **Codebook collapse:** Todos os vetores convergem para o mesmo centróide
2. **Gradientes ruidosos:** STE pode não ser preciso o suficiente
3. **Não convergir:** CromLinear pode simplesmente não aprender

## Critério de Conclusão

CromLinear atinge >90% accuracy em MNIST (nn.Linear atinge ~97%).
