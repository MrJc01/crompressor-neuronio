# Testes Sintéticos — Plano de Validação da CromLinear

*Antes de colocar CromLinear num Transformer, ela precisa provar que converge SOZINHA.*

---

## Teste 1: Regressão Linear

**Objetivo:** Verificar se CromLinear aprende uma transformação linear simples.

```
Tarefa: y = W·x + b (W é 32→16, batch=64)
Baseline: nn.Linear(32, 16) — deve convergir em <100 steps
CromLinear: CromLinear(32, 16, K=16, D=8) — aceitável em <500 steps
Critério: MSE Loss < 0.01
```

**Por que este teste importa:** Se CromLinear não aprende y=Wx, nada mais funciona.

---

## Teste 2: XOR (Não-Linear)

**Objetivo:** Verificar se MLP com CromLinear resolve tarefa não-linear clássica.

```
Tarefa: XOR(a, b) — 2 inputs, 1 output
Arquitetura: CromLinear(2, 16) → ReLU → CromLinear(16, 1) → Sigmoid
Baseline: nn.Linear equivalente — accuracy >99%
Critério: Accuracy > 95%
```

**Por que este teste importa:** Prova que CromLinear suporta composição com não-linearidades.

---

## Teste 3: MNIST (Escala Real)

**Objetivo:** Verificar se CromLinear funciona em tarefa de classificação real.

```
Tarefa: Classificar dígitos escritos à mão (28×28 → 10 classes)
Arquitetura: CromLinear(784, 256) → ReLU → CromLinear(256, 10)
Baseline: nn.Linear equivalente — accuracy ~97%
Critério: Accuracy > 90% (perda aceitável de até 7% vs baseline)
K=256, D=64
Epochs: 10
```

**Por que este teste importa:** MNIST é o "hello world" de deep learning. Se CromLinear falhar aqui, precisamos repensar a abordagem.

---

## Teste 4: Sensibilidade (K e D)

**Objetivo:** Encontrar o ponto ideal de K e D para o CromGPT.

```
Tarefa: MNIST (mesmo setup do Teste 3)
Variar K: [32, 64, 128, 256, 512]
Variar D: [16, 32, 64, 128]
Medir: accuracy, loss final, utilização do codebook (% centróides usados)
```

**Saída:** Tabela + gráfico em `resultados/lab26_sensitivity.json`

---

## Protocolo de Execução

```
Para cada teste:
1. Seed fixa: torch.manual_seed(42)
2. Treinar baseline (nn.Linear) e CromLinear com MESMOS hyperparams
3. Registrar: loss por epoch, accuracy final, tempo de treino
4. Exportar JSON para resultados/
5. Gerar gráficos de comparação
```

## Critérios de Falha

| Cenário | Ação |
|---------|------|
| Teste 1 falha (regressão) | Bug no código ou STE não funciona. Debug. |
| Teste 2 falha (XOR) | Não-linearidade interfere. Tentar Gumbel-Softmax. |
| Teste 3 < 85% (MNIST) | K muito pequeno ou D muito grande. Ajustar. |
| Codebook utilização < 30% | Codebook collapse. Adicionar commitment loss. |
