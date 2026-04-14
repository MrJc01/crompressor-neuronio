# 🔄 Prompt de Continuação — Tensor-Vivo

> Cole este prompt ao abrir um novo chat para continuar a pesquisa.

---

```
Estou continuando a pesquisa "Tensor-Vivo" do repositório crompressor-neuronio
(em /home/j/Área de trabalho/crompressor-neuronio/).

## Contexto

O projeto investiga se o motor Crompressor (CDC + Codebook + Merkle) pode
substituir diretamente os pesos/tensores de redes neurais.

## O Que Já Foi Feito (Fases 0-2 completas)

### Exp0: CDC sobre pesos reais → 0% dedup (CDC hash exato não serve para floats)
### Exp1: Codebook K-Means → K=128 B=16: 96.43% acc com 18.5x compressão
### Exp2: Codebook Learning → K=128 B=16: 97.56% acc com 5,770 params (40.8x menos)
  - K=256 B=32: 98.08% — SUPEROU baseline de 97.53%
  - Convergência em 1 epoch
  - Implementado com CodebookLinear(nn.Module) usando lookup codebook[indices]

## Veredicto Atual
O Codebook é um espaço de aprendizado viável ("LoRA do Crompressor").
Testado em MNIST MLP (784→256→128→10, 235K params). Precisa escalar.

## Estrutura
- pesquisas/tensor-vivo/ — código Python (venv em .venv/)
- pesquisas/tensor-vivo/PLANEJAMENTO.md — planejamento com resultados
- pesquisas/tensor-vivo/CONCLUSOES.md — veredicto com dados
- pesquisas/tensor-vivo/exp{0,1,2}_*/ — scripts + resultados.md
- pesquisas/tensor-vivo/dados/ — JSONs de resultados
- pesquisas/tensor-vivo/PROXIMOS_PASSOS.md — checklist do que falta

## O Que Quero Fazer Agora
[ESCREVA AQUI O QUE QUER FAZER, ex:]
- Escalar para CIFAR-10 CNN
- Implementar CDC com LSH (similaridade em vez de hash exato)
- Testar em modelo Transformer (GPT-2 small)
- Integrar CodebookLinear com o motor Go .crom
- [ou qualquer outra coisa]
```
