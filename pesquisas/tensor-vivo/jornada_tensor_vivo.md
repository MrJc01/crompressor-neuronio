# 🧬 A Jornada do Tensor-Vivo — Do Ceticismo à Validação

> *"Talvez não sirva para nada."*
> — O começo honesto de toda boa pesquisa.

---

## Capítulo 1: A Dúvida Inicial

Tudo começou com uma pergunta incômoda: **o Crompressor-Neurônio realmente serve para algo no mundo real?**

O projeto tinha ideia bonita — usar técnicas de compressão de dados (CDC, codebooks, Merkle Trees) diretamente nos pesos de redes neurais. Mas ideia bonita não paga conta. Precisávamos de evidência empírica.

A preocupação era legítima:
- CDC com hash exato já tinha **falhado** no Exp0 (zero deduplicação em pesos float32)
- A codificação DNA Base-4 nunca foi testada
- O projeto tinha muito "framework" e pouco "resultado"
- A pergunta séria era: **isso é só Vector Quantization com marketing diferente?**

Era hora de parar de construir infraestrutura e **provar ou refutar a tese**.

---

## Capítulo 2: MNIST — A Primeira Faísca (Exp1-2)

### Exp1: Quantização Pura
Começamos pelo mais simples possível: uma MLP de 235K params no MNIST.
Codebook K-Means → substituir pesos → medir accuracy.

**Resultado:** K=512 B=16 → **96.97%** accuracy (vs 97.53% baseline). Perda de apenas 0.56%.

*"Ok, isso funciona. Mas será que é só porque o MNIST é trivial?"*

### Exp2: Codebook Learning — O Momento Eureka
Aqui é onde tudo mudou. Em vez de só quantizar, **treinamos o codebook**.

Congelamos os índices e deixamos apenas os centróides como parâmetros treináveis.

**Resultado:** K=256 B=32 → **98.08%** — **SUPEROU o baseline** de 97.53%.

Espere. Leia de novo. O modelo com **13x menos parâmetros** teve accuracy **maior** que o original.

Isso não deveria acontecer em teoria pura de quantização. O que estava acontecendo era um **efeito de regularização** — o codebook forçava compartilhamento de pesos entre neurônios, e isso combatia overfitting.

**A primeira hipótese estava confirmada: o codebook não é só compressão, é um espaço de aprendizado alternativo.**

---

## Capítulo 3: CIFAR-10 CNN — Escala Funciona? (Exp3)

*"MNIST é brinquedo. Mostra que funciona de verdade."*

Passamos para CIFAR-10 com uma CNN real: Conv2d + Linear, 1M params.

Desafio novo: como quantizar **kernels convolucionais**? Resposta: flatten → chunk → K-Means. Simples e funciona.

**Resultado:** K=256 B=8 → **77.66%** (vs 77.86% baseline). Gap de apenas 0.20%.

Mas o número que importa é outro: **145.3x compressão**. De 1M params para 7,370.

E K=128 B=8 alcançou **249.1x compressão** com accuracy viável.

**Descoberta:** A compressão **escala inversamente** — quanto maior o modelo, maior a compressão possível. Isso porque modelos grandes têm mais redundância nos pesos.

---

## Capítulo 4: O Colab, a A100, e a Dor (Exp5 v1)

### A decisão de ir para Transformer
Com MLP e CNN validados, o próximo passo era óbvio: **GPT-2 Small** (124M params).

Se o codebook learning funcionasse em um Transformer, teríamos evidência forte o suficiente para um paper.

### O inferno do Colab
Migramos para o Google Colab para usar uma A100. O que parecia simples virou pesadelo:
- **Estimei 25 minutos. Demorou 3+ horas.** Erro de 8x na estimativa porque subestimei o volume de dados (67K amostras × 2093 batches × 124M params forward)
- O Colab desconectou múltiplas vezes
- Os resultados não salvavam
- Creditos queimando a $5.37 compute units/hora
- Teve que rodar várias vezes até conseguir salvar tudo no Drive

### O Bug Fantasma
Os resultados do v1 pareciam bons: **91.86%** accuracy, superando o baseline de 90.60%.

Mas algo não batia: `layers_replaced: 0` e `codebook_params: 0`.

**O script NÃO estava substituindo camada nenhuma.**

### A Descoberta: Conv1D ≠ nn.Linear

O GPT-2 do HuggingFace usa `transformers.pytorch_utils.Conv1D` — uma classe customizada que é *funcionalmente similar* a `nn.Linear` mas com peso **transposto**.

```python
# nn.Linear: weight = (out_features, in_features), forward = x @ W.T + b
# Conv1D:    weight = (in_features, out_features), forward = x @ W + b
```

O `replace_linear_with_codebook` verificava `isinstance(module, nn.Linear)` → **sempre False** para Conv1D.

**Os resultados v1 treinavam apenas biases + LayerNorm + classification head (122K params).** E mesmo assim superaram o baseline! Isso acidentalmente provou que fine-tuning mínimo (biases+LN) é extremamente eficiente para GPT-2.

Mas **não testava o codebook de jeito nenhum.**

---

## Capítulo 5: A Migração para Vast.ai

### O fim dos créditos
Colab: créditos esgotados. A pesquisa parecia parada.

Pesquisamos alternativas:
- **Vast.ai** — marketplace P2P, mais barato
- **RunPod** — meio-termo entre preço e confiabilidade
- **Lambda Labs** — bare-metal, sem disponibilidade

### A primeira vez com Vast.ai (uma saga)
Decidimos fazer tudo pelo terminal. O que parecia "5 minutos" virou uma batalha:

1. **Instalar CLI** — `pip install --break-system-packages vastai` (Linux moderno bloqueia pip global)
2. **Configurar API key** — ✅ fácil
3. **Gerar SSH key** — não existia, teve que gerar `ed25519`
4. **Registrar SSH key no Vast.ai** — bug: `vastai create ssh-key ~/.ssh/id_ed25519.pub` armazenou o *caminho* do arquivo em vez do *conteúdo*. Fix: `vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"`
5. **1ª instância (RTX 5060 Ti, China)** — ficou "running" mas SSH refused. Porta não abriu. Destruída.
6. **2ª instância (RTX 4070, New York)** — ficou "created" por 8 minutos sem iniciar. Destruída.
7. **3ª instância (RTX A4000, Polônia)** — 15+ minutos baixando o Docker image. Mas FUNCIONOU!

**Lição:** Filtrar por `reliability > 0.999`. A A4000 teve 100% reliability e custou $0.08/hr.

### SSH funcionou!
```
NVIDIA RTX A4000, 16376 MiB
SSH_OK
```

Após tanto sofrimento, esse `SSH_OK` foi lindo.

---

## Capítulo 6: Exp5 v2 — A Verdade

### O treinamento real
Baseline: 3 epochs de fine-tuning do GPT-2 no SST-2.
- Epoch 1: 89.22% (747s)
- Epoch 2: 91.06% (751s)  
- Epoch 3: 91.51% (751s) ← **nosso baseline**

**Verificação crucial:** `Tipo da camada: Conv1D (weight: torch.Size([768, 2304]))` — correto!
`Layers replaced: 48` — **TODAS as camadas alvo substituídas!**

### A quantização destrói...
Pré-treino com codebook (K=256 B=16): **50.80%** — accuracy de moeda jogada para cima.

A quantização com K=256 é brutal para 124M params. Os centróides não conseguem representar a diversidade dos pesos.

### ...mas o Learning recupera
| Epoch | Accuracy | Recovery |
|---|---|---|
| 0 (pós-quant) | 50.80% | — |
| 1 | **78.56%** | +27.75pp |
| 2 | 82.22% | +31.42pp |
| 3 | 81.19% | +30.39pp |
| 4 | 82.91% | +32.11pp |
| 5 | **83.72%** | **+32.91pp** |

**De moeda jogada (50.8%) para 83.7% apenas movendo os centróides do codebook.**

### O veredicto
```
  Arquitetura     | Baseline | Codebook | Gap    | Compressão | Recovery
  ----------------+----------+----------+--------+------------+---------
  MNIST MLP       |  97.53%  |  97.56%  | +0.03% |    40.8x   |  100.0%
  CIFAR-10 CNN    |  77.86%  |  77.66%  | -0.20% |   145.3x   |   99.7%
  GPT-2 Transf.   |  91.51%  |  83.72%  | -7.80% |   389.5x   |   91.5%
```

91.5% do baseline com **389.5x compressão**. Não é perfeito, mas é real.

---

## Capítulo 7: O Que Aprendemos

### Sobre a Tese
1. **Codebook Learning funciona em 3 arquiteturas** — não é acidente, é padrão
2. **A compressão escala** — 40x (MLP) → 145x (CNN) → 389x (Transformer)
3. **O gap também escala** — 0% (MLP) → 0.2% (CNN) → 8% (Transformer)
4. **K precisa crescer** com o modelo — K=256 é insuficiente para 124M params
5. **O efeito de regularização desaparece** em modelos grandes — não supera o baseline

### Sobre Engenharia
1. **Sempre verifique `isinstance()`** — abstrações de framework escondem tipos
2. **HuggingFace Conv1D ≠ nn.Linear** — mesma math, tipos diferentes
3. **Estimativas de tempo: calcule batches × epochs × configs** primeiro
4. **Vast.ai > Colab** para jobs de pesquisa: 10x mais barato, SSH real
5. **SSH keys: passe o conteúdo, não o caminho** ao usar CLIs
6. **`nohup`** é essencial para treinamento remoto

### Sobre Pesquisa
1. **Bugs que revelam insights** — o v1 "bugado" provou biases+LN fine-tuning
2. **Resultados parciais têm valor** — 91.5% recovery não é 99%, mas é publicável
3. **Custos importam** — $0.50 no Vast.ai vs $16+ no Colab para o mesmo resultado
4. **Falhar rápido, documentar tudo** — cada falha nesta sessão gerou aprendizado

---

## Capítulo 8: Para Onde Vamos

### O gap de 8% é fechável?
Provavelmente sim, com:
- **Mais centróides** (K=1024, K=2048) — dá mais expressividade
- **Mais epochs** (10-20) — o loss ainda estava caindo no epoch 5
- **Codebook per-layer** — em vez de K fixo, adaptar K pela variância da camada
- **Residual codebook** — quantizar, treinar, re-quantizar os resíduos
- **Mixed precision** — manter embeddings e heads em full precision

### O que isso habilita
Se fecharmos para 95%+ do baseline:
1. **Codebook-as-LoRA** — adaptar LLMs distribuindo apenas codebooks (KB, não GB)
2. **Formato .crom** — binário eficiente para codebook + índices
3. **FUSE driver** — servir modelos codebook-quantized via filesystem virtual
4. **Paper acadêmico** — "Learnable Codebook Quantization as Parameter-Efficient Fine-Tuning"

---

## Os Números que Importam

| Métrica | Valor |
|---|---|
| Arquiteturas testadas | 3 (MLP, CNN, Transformer) |
| Melhor compressão | 389.5x (GPT-2) |
| Melhor recovery | 100.0% (MNIST — superou baseline) |
| Custo total Vast.ai | ~$0.50 |
| Custo Colab desperdiçado | ~$16+ |
| Bug mais caro | Conv1D vs nn.Linear |
| Insght acidental mais valioso | biases+LN fine-tuning = 91.86% no GPT-2 |

---

> *"Talvez não sirva para nada"* virou *"Validado em 3 arquiteturas."*
>
> O neurônio que comprime **é** o neurônio que pensa.
> E o codebook **é** a memória comprimida desse pensamento.
