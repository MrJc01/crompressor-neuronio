# 🔬 Tensor-Vivo — Planejamento & Checklist

> **Status:** ✅ Fases 0-2 concluídas com resultados positivos
> **Última atualização:** 2026-04-14
> **Tese:** O Crompressor (CDC + Codebook DNA + Merkle) pode substituir diretamente os pesos numéricos de tensores em redes neurais.

## 📊 Resultados Resumidos

| Hipótese | Resultado | Status |
|---|---|---|
| H1: CDC hash exato encontra dedup em pesos | 0% dedup | ❌ Refutada |
| H2: Codebook K-Means preserva accuracy | 96.97% com 9.4x compressão | ✅ Confirmada |
| H3: Codebook Learning funciona | **98.08% — SUPEROU baseline** | ✅✅ Confirmada fortemente |
| Melhor config | K=128 B=16: **97.56% com 5,770 params** (40.8x menos) | 🏆 |

> Ver resultados detalhados em `CONCLUSOES.md`

---

## 🧠 Contexto e Estado da Arte

A pesquisa revelou que a ideia NÃO é esquizofrenia — ela tem nome no mundo
acadêmico: **Vector Quantization (VQ) de Pesos de Redes Neurais**. Porém, o
que você está propondo tem uma diferença crucial:

| Aspecto | VQ Tradicional | Tensor-Vivo (Crompressor) |
|---|---|---|
| Chunking | Fixo (blocos de N floats) | **CDC** (content-defined, tamanho variável) |
| Codebook | K-means sobre floats | **DNA Base-4** com Codebook treinável |
| Integridade | Nenhuma | **Merkle Tree** por chunk |
| Dedup | Não existe | **Dedup hash** (chunks idênticos → 1 ref) |
| Base teórica | Teoria da Informação + Clustering | **Shannon + Compressão = Cognição** |

### Papers Relevantes (2025-2026)
- **FVQ/VQBridge (2025):** Treina codebooks com 100% utilização, sem collapse
- **LooC (2026):** VQ em low-dimensional "slots" (similar a chunking sub-vetor)
- **Meta-Quantization (2025):** Hyper-network que gera codebook dinamicamente
- **Codebook Features (ICML 2025):** Hidden states quantizados em codebook = interpretabilidade
- **CSVQ (2025):** Codebook compartilhado escalar por canal = memória linear

### O Que Torna o Crompressor Diferente
1. **CDC define os limites do chunk pela semântica dos dados**, não por tamanho fixo
2. **DNA encoding** potencialmente captura padrões que float32 não captura
3. **Merkle** garante que nenhum peso foi corrompido (verificabilidade)
4. **Dedup** entre neurônios: se dois neurônios têm pesos similares, compartilham chunk

---

## 🎯 O Que Vamos Provar (Hipóteses)

### H1: Preservação de Estrutura
> Chunks CDC de pesos float preservam a geometria do espaço de pesos?
> I.e., neurônios "parecidos" no espaço de pesos geram chunks "parecidos" no espaço CDC?

### H2: Roundtrip Viável
> Comprimir pesos → DNA Codebook → Lookup → Forward pass:
> a accuracy é aceitável (>90% da original)?

### H3: Codebook Aprendível
> Se congelarmos os chunks (estrutura) e permitirmos só
> modificar as entries do Codebook, o modelo consegue "aprender"?
> (Isso seria o "LoRA do Crompressor")

### H4: Dedup = Redundância Neural
> A taxa de deduplicação CDC dos pesos correlaciona com
> redundância funcional dos neurônios?
> (Neurônios duplicados no CDC = neurônios que podem ser podados?)

---

## 🏗️ Arquitetura dos Experimentos

```
pesquisas/tensor-vivo/
├── README.md                    # Visão geral (✅ feito)
├── PLANEJAMENTO.md              # Este arquivo
├── exp0_analise_estrutural/     # Fase 0: CDC sobre pesos reais
│   ├── extract_weights.py       # Extrai pesos de modelo PyTorch
│   ├── cdc_analysis.go          # Processa com FastCDC
│   └── resultados.md
├── exp1_roundtrip/              # Fase 1: Compress→Decompress→Accuracy
│   ├── quantize_codebook.py     # Mapeia pesos→codebook entries
│   ├── inference_test.py        # Roda forward pass com pesos codebook
│   └── resultados.md
├── exp2_codebook_learning/      # Fase 2: Fine-tune só o codebook
│   ├── train_codebook.py        # Training loop com STE
│   └── resultados.md
├── exp3_dedup_pruning/          # Fase 3: Dedup CDC como pruning
│   ├── dedup_correlation.py
│   └── resultados.md
└── dados/                       # Outputs dos experimentos
```

---

## 📐 Stack Técnico

| Componente | Tecnologia | Razão |
|---|---|---|
| Modelo de teste | PyTorch (MNIST MLP, depois CIFAR CNN) | Simples, rápido, mensurável |
| Extração de pesos | Python (torch.save → .bin) | Nativo do PyTorch |
| CDC Chunking | Go (fastcdc-go) **ou** Python (fastcdc) | Já temos em Go, mas Python simplifica integração |
| Codebook | Python (sklearn KMeans + custom DNA) | Precisamos integrar com PyTorch |
| Forward pass | PyTorch (custom nn.Module) | Straight-Through Estimator |
| Métricas | Python (accuracy, perplexity, compression ratio) | Padrão |

> **Decisão de design:** Para máxima velocidade, vamos fazer tudo em Python
> nesta fase exploratória. O CDC em Go fica para a versão de produção.
> Python tem `fastcdc` package que faz a mesma coisa.

---

## 🔬 Fases de Execução

### Fase 0: Análise Estrutural (~3-4h)
**Pergunta:** Como os pesos de uma rede neural se comportam quando
processados pelo FastCDC?

**Experimento:**
1. Treinar MLP simples no MNIST (2 camadas, ~100K params)
2. Extrair todos os pesos como array de bytes (float32 → raw bytes)
3. Rodar FastCDC sobre esses bytes
4. Medir:
   - Quantos chunks CDC são gerados?
   - Qual a taxa de deduplicação entre camadas?
   - Qual a entropia de Shannon de cada chunk?
   - Chunks de embedding vs linear: padrões diferentes?
5. Visualizar: histograma de tamanhos de chunk, mapa de dedup

**Critério de sucesso:** Dedup > 0% (prova que existe redundância
estrutural nos pesos que CDC captura)

### Fase 1: Roundtrip & Forward Pass (~4-6h)
**Pergunta:** Se comprimirmos os pesos com codebook e descomprimirmos,
o modelo ainda funciona?

**Experimento:**
1. Pegar o MLP treinado da Fase 0
2. Agrupar weights em chunks CDC
3. Para cada chunk único, criar uma entry no Codebook (centróide via k-means)
4. Substituir cada peso original pelo centróide mais próximo do codebook
5. Rodar inference com os pesos quantizados
6. Medir accuracy: original vs codebook-quantized
7. Variar K (tamanho do codebook): 16, 64, 256, 1024
8. Plotar: accuracy vs compression ratio vs K

**Critério de sucesso:** Com K=256, accuracy > 95% da original

### Fase 2: Codebook Learning (~4-6h)
**Pergunta:** Se congelarmos a estrutura (chunk→index mapping) e
só treinarmos os valores do Codebook, o modelo aprende?

**Experimento:**
1. Inicializar codebook com centróides da Fase 1
2. Congelar o mapeamento chunk→index (cada peso sabe qual entry usar)
3. Training loop:
   - Forward: weight = codebook[index[i]] (lookup)
   - Backward: gradiente via Straight-Through Estimator (STE)
   - Update: só o codebook recebe gradient descent
4. Treinar 10 epochs, medir accuracy epoch-by-epoch
5. Comparar com LoRA (mesmo nº de params treináveis)

**Se funcionar:** Isso prova que o Codebook do Crompressor é um espaço
de aprendizado viável — análogo ao LoRA, mas no espaço comprimido.

**Critério de sucesso:** Accuracy recupera para >98% em 10 epochs

### Fase 3: Dedup como Insight Neural (~2-3h)
**Pergunta:** A deduplicação CDC revela algo sobre a estrutura
interna da rede?

**Experimento:**
1. Pegar modelo CNN treinado (CIFAR-10)
2. Extrair pesos de CADA camada separadamente
3. Rodar CDC e medir dedup intra-camada e inter-camada
4. Correlacionar dedup rate com:
   - Magnitude dos pesos (camadas com pesos menores = mais dedup?)
   - Sensibilidade ao pruning (camadas que podem ser podadas = mais dedup?)
   - Posição no modelo (primeiras camadas vs últimas)
5. Testar: se merge chunks duplicados (weight sharing), accuracy cai?

**Se funcionar:** Prova que CDC dedup é uma forma de análise
de redundância neural que não precisa de gradientes.

**Critério de sucesso:** Correlação > 0.5 entre dedup rate e prunability

---

## ✅ Checklist Detalhado

### Setup Inicial
- [ ] Criar venv Python: `python -m venv pesquisas/tensor-vivo/.venv`
- [ ] Instalar deps: `pip install torch torchvision numpy matplotlib scikit-learn`
- [ ] Instalar fastcdc Python: `pip install fastcdc`
- [ ] Verificar que PyTorch funciona: `python -c "import torch; print(torch.__version__)"`
- [ ] Criar `pesquisas/tensor-vivo/requirements.txt`

### Fase 0: Análise Estrutural
- [ ] Criar `exp0_analise_estrutural/`
- [ ] `extract_weights.py`:
  - [ ] Treinar MLP no MNIST (784→256→128→10)
  - [ ] Salvar modelo treinado (.pt)
  - [ ] Extrair weights de cada camada como bytes (float32.tobytes())
  - [ ] Salvar como .bin para cada camada
  - [ ] Imprimir accuracy baseline do modelo
- [ ] `cdc_analysis.py` (CDC em Python):
  - [ ] Ler cada .bin de pesos
  - [ ] Rodar FastCDC (min=64, avg=512, max=4096)
  - [ ] Calcular hash SHA-256 de cada chunk
  - [ ] Contar chunks únicos vs duplicados por camada
  - [ ] Calcular entropia de Shannon por chunk
  - [ ] Salvar resultados em `dados/exp0_results.json`
- [ ] Visualização:
  - [ ] Histograma de tamanhos de chunk
  - [ ] Heatmap de dedup inter-camada
  - [ ] Box plot de entropia por camada
  - [ ] Salvar plots em `dados/exp0_*.png`
- [ ] `resultados.md`:
  - [ ] Tabela: camada × chunks × dedup × entropia
  - [ ] Conclusão: CDC revela estrutura nos pesos? Sim/Não

### Fase 1: Roundtrip & Forward Pass
- [ ] Criar `exp1_roundtrip/`
- [ ] `quantize_codebook.py`:
  - [ ] Carregar modelo treinado da Fase 0
  - [ ] Para cada camada, achatar pesos em blocos de N floats
  - [ ] Rodar K-means para criar codebook de K centróides
  - [ ] Substituir cada bloco pelo centróide mais próximo
  - [ ] Reconstruir tensor de pesos a partir do codebook
  - [ ] Injetar pesos quantizados no modelo
- [ ] `inference_test.py`:
  - [ ] Carregar modelo com pesos originais → medir accuracy
  - [ ] Carregar modelo com pesos codebook K=16 → accuracy
  - [ ] K=64 → accuracy
  - [ ] K=256 → accuracy
  - [ ] K=1024 → accuracy
  - [ ] Calcular compression ratio para cada K
  - [ ] Plotar curva: accuracy vs K vs compression ratio
  - [ ] Salvar em `dados/exp1_results.json`
- [ ] `resultados.md`:
  - [ ] Tabela: K × accuracy × ratio × perda
  - [ ] Gráfico embeddado
  - [ ] Conclusão: qual K mínimo preserva accuracy?

### Fase 2: Codebook Learning
- [ ] Criar `exp2_codebook_learning/`
- [ ] `train_codebook.py`:
  - [ ] Implementar `CodebookLinear(nn.Module)`:
    - [ ] `__init__`: recebe codebook (K×D), indices (N,)
    - [ ] `forward`: weight = codebook[indices], retorna F.linear(x, weight)
    - [ ] Usar Straight-Through Estimator no backward
  - [ ] Substituir cada Linear do MLP por CodebookLinear
  - [ ] Congelar indices (não treinável)
  - [ ] Codebook.requires_grad = True
  - [ ] Training loop: 10 epochs, lr=1e-3, Adam
  - [ ] Log accuracy por epoch
  - [ ] Comparar nº de params treináveis: codebook vs LoRA equivalente
  - [ ] Salvar em `dados/exp2_results.json`
- [ ] `resultados.md`:
  - [ ] Curva accuracy por epoch
  - [ ] Comparação com baseline (full weights) e LoRA
  - [ ] Conclusão: Codebook é um espaço de aprendizado viável?

### Fase 3: Dedup como Insight Neural
- [ ] Criar `exp3_dedup_pruning/`
- [ ] `dedup_correlation.py`:
  - [ ] Treinar CNN no CIFAR-10 (Conv2d × 3 + Linear × 2)
  - [ ] Extrair pesos de cada camada
  - [ ] CDC + dedup rate por camada
  - [ ] Calcular magnitude média dos pesos por camada
  - [ ] Pruning sensitivity test: zerar cada camada e medir accuracy drop
  - [ ] Correlação: dedup_rate vs accuracy_drop
  - [ ] Salvar em `dados/exp3_results.json`
- [ ] `resultados.md`:
  - [ ] Scatter plot: dedup_rate vs prunability
  - [ ] Correlação de Pearson
  - [ ] Conclusão: CDC dedup prediz redundância neural?

### Relatório Final
- [ ] Criar `pesquisas/tensor-vivo/CONCLUSOES.md`:
  - [ ] Resumo das 4 fases
  - [ ] Resposta a cada hipótese (H1-H4)
  - [ ] "O Crompressor como neurônio funciona?" → Sim/Não/Parcialmente
  - [ ] Próximos passos se funcionar
  - [ ] Paper draft outline (se resultados forem publicáveis)

---

## ⚠️ Riscos

| Risco | Impacto | Mitigação |
|---|---|---|
| CDC não encontra dedup em pesos (H1 falha) | Alto | Normal para modelos pequenos. Testar com modelos maiores. |
| Quantização por codebook destrói accuracy | Alto | Aumentar K. Se K=1024 não funcionar, a granularidade do codebook é insuficiente. |
| STE não converge no Codebook Learning | Médio | Usar Gumbel-Softmax em vez de STE. Paper FVQ tem soluções. |
| Python fastcdc se comporta diferente do Go | Baixo | Verificar com dados sintéticos antes. |
| MNIST muito simples pra mostrar dedup | Médio | É o mínimo viável. Se funcionar, escalar para CIFAR/GPT-2-small. |

---

## 📅 Estimativa de Tempo

| Fase | Tempo | Dependência |
|---|---|---|
| Setup | 30min | Nenhuma |
| Fase 0 | 3-4h | Setup |
| Fase 1 | 4-6h | Fase 0 |
| Fase 2 | 4-6h | Fase 1 |
| Fase 3 | 2-3h | Fase 0 |
| Relatório | 1-2h | Todas |
| **Total** | **~15-22h** | |

> Fase 3 pode rodar em paralelo com Fase 1/2 (é independente após Fase 0).

---

> **Próximo passo:** Setup + Fase 0 (extrair pesos, rodar CDC, ver se existe estrutura).
> Se a Fase 0 mostrar dedup > 0%, a tese tem pernas. Se não, repensamos.
