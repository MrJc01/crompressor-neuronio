# Exp1 — Resultados: Roundtrip com Codebook K-Means

## Dados Brutos

**Baseline:** 97.53% accuracy (MLP MNIST, 235K params)

### Block Size = 16 floats (64 bytes)
| K | Accuracy | Perda | Compressão | MSE |
|---|---|---|---|---|
| 4 | 15.93% | -81.60% | 31.2x | 0.003474 |
| 8 | 29.64% | -67.89% | 30.4x | 0.003017 |
| 16 | 57.79% | -39.74% | 29.0x | 0.002518 |
| 32 | 80.99% | -16.54% | 26.5x | 0.001913 |
| 64 | 93.14% | -4.39% | 22.6x | 0.001130 |
| **128** | **96.43%** | **-1.10%** | **18.5x** | **0.000808** |
| 256 | 96.31% | -1.22% | 14.0x | 0.000672 |
| **512** | **96.97%** | **-0.56%** | **9.4x** | **0.000525** |

### Block Size = 32 floats (128 bytes)
| K | Accuracy | Perda | Compressão | MSE |
|---|---|---|---|---|
| 64 | 87.07% | -10.46% | 26.0x | 0.001085 |
| 128 | 92.55% | -4.98% | 17.9x | 0.000937 |
| 256 | 96.00% | -1.53% | 11.0x | 0.000764 |
| 512 | 96.73% | -0.80% | 6.2x | 0.000549 |

### Block Size = 64 floats (256 bytes)
| K | Accuracy | Perda | Compressão | MSE |
|---|---|---|---|---|
| 128 | 92.68% | -4.85% | 12.0x | 0.000865 |
| 256 | 95.47% | -2.06% | 6.5x | 0.000608 |
| 512 | 96.60% | -0.93% | 3.4x | 0.000260 |

### Block Size = 128 floats (512 bytes)
| K | Accuracy | Perda | Compressão | MSE |
|---|---|---|---|---|
| 256 | 94.29% | -3.24% | 3.5x | 0.000312 |
| 512 | 96.55% | -0.98% | 2.3x | 0.000235 |

## Configurações Destaque

### 🏆 Melhor Accuracy com Alta Compressão
- **K=128, Block=16:** 96.43% (-1.10%) com **18.5x** compressão

### 🏆 Melhor Accuracy Absoluta
- **K=512, Block=16:** 96.97% (-0.56%) com 9.4x compressão

### 🏆 Máxima Compressão Viável (>90%)
- **K=64, Block=16:** 93.14% (-4.39%) com **22.6x** compressão

## Conclusão

> **A TESE SE SUSTENTA.**

Os pesos de uma rede neural treinada **podem ser representados por um 
codebook de K centróides** com perda mínima de accuracy:

- Com **128 entradas de codebook** e blocos de 16 floats, mantivemos
  **96.43%** accuracy (vs 97.53% original), com **18.5x compressão**.
- Isso significa que ~235K parâmetros foram representados efetivamente
  por ~128 centróides × 16 dims = ~2048 floats + índices.

**O Codebook É a Representação.** Os pesos não precisam existir como
float32 individuais — eles podem ser referências a entradas de um
dicionário semântico comprimido.

## Próximo Passo

**Exp2:** Se congelarmos os índices (qual centróide cada bloco usa)
e treinarmos APENAS os valores do codebook, o modelo recupera accuracy?
Isso provaria que o codebook é um **espaço de aprendizado viável**.
