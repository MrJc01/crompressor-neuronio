# Crompressor: Pesquisa 3 — Papel 0
## Estudo Comparativo de Compressão Híbrida (PTQ) vs Quantization Aware Training (QAT)

**Data:** Abril de 2026
**Objetivo:** Avaliar o impacto da *Post-Training Quantization* (PTQ) seletiva nas camadas Feed-Forward (FFN) em modelos SOTA (State of the Art) como o Phi-3 / Llama-3, visando inferência viável em hardwares de memória restrita (Edge - 8GB RAM).

---

### 1. Introdução e Diagnóstico Forense
Após o sucesso do formato CromLinear no treinamento nativo (QAT) documentado na Pesquisa 2, a transposição direta do algoritmo K-Means agressivo (K=256) sobre pesos pré-treinados (PTQ global) gerou um colapso severo no raciocínio do modelo (*Semantic Drift*).
O diagnóstico evidenciou que esmagar as delicadas matrizes de Atenção e o `lm_head` destrói as distribuições de probabilidade do modelo. Além disso, a tentativa de processar as inferências em CPU com um Kernel C++ puro gerou gargalos de alocação que não puderam competir com o backend multithreaded nativo do PyTorch.

### 2. Metodologia: Caminho B (Híbrido)
Para resolver a questão da degradação inteligente e garantir viabilidade em Edge Devices:
1. **Compressão Seletiva:** A quantização Crom (Codebooks) foi aplicada estritamente sobre as matrizes gigantes da arquitetura FFN (`up_proj`, `down_proj`, `gate_proj`).
2. **Preservação QI:** As matrizes de Attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) e a cabeça de linguagem (`lm_head`) foram mantidas em seu estado FP16 natural.
3. **Reconstrução On-The-Fly:** Remoção total das dependências nativas C++ para aproveitar a paralelização inerente do OneBLAS via `W_q = codebook[indices]` injetado dinamicamente no `F.linear`.

### 3. Execução da Compressão (Cloud)
*A preencher após a execução na Vast.ai.*
- Máquina Utilizada: 
- Tempo de Extração (K-Means com FAISS):
- Redução de RAM do Modelo (Original vs Crom): 

### 4. Benchmarking Local (Edge - 8GB RAM)
*A preencher após a geração.*
- **Time to First Token (TTFT):** ...
- **Tokens/s (Geração Autoregressiva):** ...
- **Memória Alocada no Pico:** ...

### 5. Avaliação Qualitativa
*A preencher após o teste no `local_chat.py`.*
- O modelo consegue manter coesão em português longo?
- Existe alucinação induzida pelos erros do FFN quantizado?

### 6. Conclusão Preliminar
*(Concluir se o Caminho B é viável ou se precisaremos descartar o PTQ e voltar para o treinamento nativo na nuvem).*
