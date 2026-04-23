# CONCLUSÕES — Pesquisa 2: CromGPT

> **Status:** ⏳ Pesquisa em andamento. Este arquivo será preenchido conforme os resultados forem obtidos.

---

## Hipóteses a Validar

| # | Hipótese | Status | Resultado |
|:--|:---------|:-------|:----------|
| H1 | CromLinear converge em tarefas sintéticas (regressão, XOR, MNIST) | ✅ Parcial | XOR 100%, MNIST 95.21% (gap 2.8%). Regressão pura oscila. |
| H2 | CromGPT (Transformer com CromLinear) treina sem divergir | ⏳ | — |
| H3 | CromGPT gera texto coerente em Português | ⏳ | — |
| H4 | CromGPT ocupa menos espaço em disco que baseline nn.Linear | ⏳ | — |
| H5 | CromGPT mantém perplexidade aceitável vs baseline | ⏳ | — |
| H6 | Formato .crom v3 serializa e deserializa modelo completo | ⏳ | — |

---

## Veredictos

### Eixo 02 — CromLinear ✅ (2026-04-23)

**Veredicto: A CromLinear FUNCIONA para classificação.**

Dados empíricos:

| Teste | CromLinear | Baseline (nn.Linear) | Gap |
|-------|-----------|---------------------|-----|
| XOR | **100%** | 100% | 0% |
| MNIST | **95.21%** | 98.05% | **2.8%** |
| Compressão (MNIST layer 1) | — | — | **11.2x** |
| Codebook utilização | **100%** | — | Zero collapse |

**Achado importante:** CromLinear tem dificuldade com regressão pura (representar W arbitrária exata), mas funciona bem em classificação de alta dimensão. Isso é consistente com a literatura de VQ-VAE: codebooks são mais eficientes quando os dados têm estrutura exploitável.

**Implicação para CromGPT:** O next-token prediction é uma tarefa de classificação (classificar qual token vem depois), não regressão. Isso é um **sinal positivo** para o CromGPT.

### Eixo 03 — CromGPT  
> Veredicto pendente.

### Eixo 04 — Avaliação
> Veredicto pendente.

---

## Lições Aprendidas

1. **STE funciona mas precisa de codebook loss:** O gradiente STE sozinho não atualiza o codebook. Precisamos de `codebook_loss = MSE(quantized, continuous.detach())` para que o codebook receba gradientes.
2. **Codebook collapse NÃO ocorreu:** Com commitment loss + codebook loss, 100% dos centróides foram usados em todos os testes. O anti-collapse funcionou.
3. **LR separado ajuda:** O codebook precisa de learning rate mais alta que os pesos contínuos para convergir mais rápido.
4. **Classificação > Regressão:** CromLinear é melhor em tarefas onde a saída é uma distribuição sobre classes do que em tarefas de mapeamento exato.

---

## Conexão com Pesquisa 3

*(Definido após conclusão da Pesquisa 2)*
