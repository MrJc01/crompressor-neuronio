# PLANEJAMENTO — Pesquisa 3: CromPTQ e CPU Inference (LLMs Nativos Localmente)

> **Objetivo:** Aplicar a tecnologia CromLinear via *Post-Training Quantization* (PTQ) em modelos SOTA (State of the Art) de código aberto (Llama-3 8B ou Phi-3), reduzindo a RAM necessária de 16GB para ~2GB, e desenvolver Kernels C++ de CPU para garantir inferência em tempo real em máquinas sem GPU.
>
> **Data:** Abril 2026
> **Status:** 🔄 PLANEJAMENTO INICIAL — 0/41 items 
> **Pré-requisito:** pesquisa2 (100% concluída - Validação Empírica CromLinear e formato .cromv3)

---

## 🎯 Painel de Especialistas Simulados (Top 5)

| # | Papel | Contribuição |
|:--|:------|:-------------|
| 1 | **Arquiteto de Inferência Edge** | Estratégias para rodar LLMs de 8B em 8GB de RAM. Foco em CPU. |
| 2 | **Especialista em C++ / AVX (SIMD)** | Otimização extrema de loops, cache locality e uso de registradores vetoriais do processador para acelerar o lookup de codebooks sem depender de GPU. |
| 3 | **Pesquisador de Quantização (PTQ)** | Algoritmos de extração de centroides (K-Means), controle de degradação de matrizes pré-treinadas (outliers, ativações não lineares). |
| 4 | **SRE / Cloud Ops (Vast.ai)** | Gestão da memória (evitar OOM) ao abrir um modelo de 16GB na nuvem para realizar a compressão. Pipeline de exportação. |
| 5 | **Engenheiro HuggingFace** | Manipulação do `transformers`, extração do `state_dict`, formatação de prompt (Chat Templates) do Llama-3/Phi-3. |

### Melhores Práticas Consolidadas
1. **Delegar RAM Pesada para a Nuvem:** Máquinas de 8GB não conseguem abrir tensores FP16 de 16GB. O script PTQ deve rodar na nuvem (Vast.ai com 64GB de RAM).
2. **K-Means Ultrarrápido:** Usar `faiss-cpu` ou `faiss-gpu` na nuvem em vez do `scikit-learn` para calcular os K-Means das matrizes, economizando horas de processamento.
3. **C++ PyBind11:** Integrar o Kernel C++ via `torch.utils.cpp_extension` (JIT) para não precisarmos sair do ecossistema PyTorch no script do usuário final.
4. **Respeitar os Outliers:** Em LLMs SOTA, 1% dos pesos controlam os outliers cruciais. Talvez manter tensores de LayerNorm em FP32/FP16 puros.

### Armadilhas Identificadas
- ⚠️ **K-Means muito agressivo destrói o modelo:** Se tentarmos 64x de compressão bruta (K=256, D=64) em pesos pré-treinados, a inteligência do GPT-4 vai descer pro nível de um papagaio. Teremos que ajustar D (para 8 ou 16) no PTQ.
- ⚠️ **Kernel C++ Lento:** Fazer `for` loops em C++ sem #pragma OMP ou AVX será tão lento quanto Python. O kernel precisa ser explícito sobre vetorização.
- ⚠️ **Formato de Prompt:** Modelos SOTA quebram se não usarmos o token exato de `<|im_start|>` ou `<|eot_id|>`.

---

## Convenções

> - `[x]` = Completo
> - `[/]` = Em progresso
> - `[ ]` = Não iniciado
> - **[P1]** = Prioridade Alta (Caminho Crítico)
> - **[P2]** = Prioridade Média (Otimização)
> - **[P3]** = Prioridade Baixa (Fru-fru / Papelada)

---

## ═══════════════════════════════════════════════════════════
## EIXO 01 — PREPARAÇÃO DO MODELO ALVO E ALGORITMO PTQ
## ═══════════════════════════════════════════════════════════

### 1.1 Seleção e Mapeamento do Modelo
- [ ] **1.1.1** [P1] Selecionar o modelo: `microsoft/Phi-3-mini-4k-instruct` (3.8B, mais leve) ou `meta-llama/Meta-Llama-3-8B-Instruct` (Extremamente capaz).
- [ ] **1.1.2** [P1] Inspecionar a arquitetura da rede (nome das chaves das camadas lineares `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- [ ] **1.1.3** [P1] Validar quanta RAM é necessária para carregar o FP16 bruto no PyTorch.

### 1.2 Algoritmo de Compressão (K-Means)
- [x] **1.2.1** [P1] Escrever o compressor `ptq_compressor.py`.
- [x] **1.2.2** [P1] Implementar extração bloco a bloco (dimensão `D`) para agrupar pesos.
- [x] **1.2.3** [P1] Integrar o `faiss` (K-Means cluster) para encontrar os melhores Codebooks.
- [x] **1.2.4** [P1] Reconstruir a matriz comprimida (Loss MSE) e documentar o erro médio da quantização.
- [ ] **1.2.5** [P2] Testar valores menores de `D` (ex: D=8 ou D=16) para garantir retenção de QI do modelo.

---

## ═══════════════════════════════════════════════════════════
## EIXO 02 — VAST.AI PIPELINE (NUVEM)
## ═══════════════════════════════════════════════════════════

### 2.1 Execução da Compressão
- [ ] **2.1.1** [P1] Escrever script de provisionamento Vast.ai rápido (instância com 32GB+ RAM).
- [ ] **2.1.2** [P1] Fazer upload do `ptq_compressor.py`.
- [ ] **2.1.3** [P1] Fazer o download do modelo SOTA (HuggingFace) dentro do Vast.ai.
- [ ] **2.1.4** [P1] Executar a compressão PTQ em nuvem.
- [ ] **2.1.5** [P1] Salvar tudo usando a função atualizada `save_cromv3` modificada para arquiteturas SOTA.

### 2.2 Exportação
- [ ] **2.2.1** [P1] Fazer o SCP do arquivo gigante para a máquina local (arquivo .cromv3).
- [ ] **2.2.2** [P1] Destruir a instância Vast.ai para conter custos.

---

## ═══════════════════════════════════════════════════════════
## EIXO 03 — CROM CPU KERNEL (C++ EXTREMO)
## ═══════════════════════════════════════════════════════════

### 3.1 Kernel C++ Nativo (Sem GPU)
- [x] **3.1.1** [P1] Criar pasta `pesquisa3/kernels`.
- [x] **3.1.2** [P1] Escrever `crom_linear_cpu.cpp`: Receber `X` (vetor FP32), `Codebook` (FP16), e `Indices` (uint16_t).
- [x] **3.1.3** [P1] Fazer a reconstrução implícita e a soma matemática em um único loop em C++ (`O(N)` super rápido).
- [x] **3.1.4** [P2] Adicionar diretivas de paralelismo `#pragma omp parallel for` para usar múltiplos núcleos do processador (Intel/AMD).

### 3.2 Binding Python (JIT)
- [x] **3.2.1** [P1] Escrever `setup.py` ou carregador Just-in-Time usando `torch.utils.cpp_extension`.
- [x] **3.2.2** [P1] Criar a classe autograd `CromLinearCPUFunction` que chama a função C++ em vez do `index_select` lerdo do PyTorch.

---

## ═══════════════════════════════════════════════════════════
## EIXO 04 — INFERÊNCIA LOCAL (A MÁGICA EM 8GB)
## ═══════════════════════════════════════════════════════════

### 4.1 Script de Chat
- [x] **4.1.1** [P1] Escrever `local_chat.py`.
- [x] **4.1.2** [P1] Implementar `load_cromv3_sota()` que instancia o Llama-3/Phi-3 vazio e injeta os nossos Codebooks e Índices, aplicando o nosso Kernel C++.
- [x] **4.1.3** [P1] Carregar o Tokenizer oficial da HF (`AutoTokenizer`).
- [x] **4.1.4** [P2] Inserir `apply_chat_template` padrão da Meta/Microsoft para que o modelo entenda os turnos "system", "user" e "assistant".
- [x] **4.1.5** [P1] Gerenciar geração autoregressiva com KV Cache para não recalcular a história a cada token (essencial para velocidade).

### 4.2 Benchmark Local
- [ ] **4.2.1** [P1] Monitorar RAM durante a carga (deve cravar em <3GB).
- [ ] **4.2.2** [P1] Medir tempo do primeiro token (Time to First Token - TTFT).
- [ ] **4.2.3** [P1] Medir tokens gerados por segundo usando a CPU. Alvo: > 10 tok/s.

---

## ═══════════════════════════════════════════════════════════
## EIXO 05 — O PAPEL 3 (O Grand Finale)
## ═══════════════════════════════════════════════════════════

### 5.1 Redação do `papel3.md`
- [ ] **5.1.1** [P1] Título, Abstract e Introdução (PTQ vs Treino).
- [ ] **5.1.2** [P1] Metodologia do FAISS K-Means e preservação da perplexidade (MSE).
- [ ] **5.1.3** [P1] Análise do Kernel C++ e como a velocidade saltou em relação ao `index_select` padrão.
- [ ] **5.1.4** [P1] Resultados empíricos de Compressão: Quanto o modelo de 16GB encolheu? Coube na RAM de 8GB?
- [ ] **5.1.5** [P1] Análise Qualitativa: O modelo sabe responder raciocínio lógico (GPT-4 level)? O PTQ o lobotomizou ou não?

---

## Resumo de Items

| Eixo | Items | Prioridade | Status |
|:-----|:------|:-----------|:-------|
| 01 — Compressor PTQ | 8 | P1 ⭐ | [ ] |
| 02 — Vast Pipeline | 7 | P1 | [ ] |
| 03 — C++ CPU Kernel | 6 | P1 ⭐ | [ ] |
| 04 — Inferência Local | 8 | P1/P2 | [ ] |
| 05 — Papel 3 | 5 | P1 | [ ] |
| **TOTAL** | **~34 items** | | |
