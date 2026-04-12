# 📄 Referências Acadêmicas — Crompressor-Neurônio

> Papers e artigos relevantes (2024-2026) organizados por área de pesquisa.
> Cada referência inclui relevância direta para o projeto.

---

## 1. Vector Quantization & Compressão de Modelos

### LLVQ — Leech Lattice Vector Quantization for LLMs
- **Data:** Março 2026
- **Fonte:** arXiv
- **Resumo:** Usa a lattice Leech (empacotamento ótimo em 24 dimensões) para VQ de LLMs, eliminando a necessidade de codebooks massivos materializados. Executa busca via extended Golay code.
- **Relevância:** Base teórica para nosso VQ Neural (doc 04). Prova que VQ pode ser feita de forma eficiente sem codebooks gigantes — exatamente como nosso Codebook DNA.
- **Aplicação:** Implementar Leech lattice como alternativa ao codebook treinável para casos de alta dimensionalidade.

### TurboQuant — Online Vector Quantization for KV Cache
- **Data:** 2025 (Aceito ICLR 2026)
- **Fonte:** arXiv / ICLR
- **Resumo:** Compressão data-oblivious e training-free do KV cache durante inferência. Redução de 6x na memória do KV cache.
- **Relevância:** Aplicável ao nosso cache de ativações (Forward Pass Diferencial). Podemos aplicar TurboQuant sobre os vetores latentes cacheados por hash CDC.
- **Aplicação:** Comprimir activation cache do neurônio fixo.

### UniQL — Unified Quantization + Low-Rank Compression
- **Data:** Início 2026
- **Fonte:** arXiv
- **Resumo:** Framework que combina post-training quantization com low-rank decomposition. Configurável para diferentes níveis de memória/computação.
- **Relevância:** Modelo de referência para nosso sistema de compressão adaptativa (Vertente Semi-Fixa).
- **Aplicação:** Inspiração para o mecanismo de atualização parcial via SVD + quantização.

---

## 2. Delta Fine-Tuning & Adapters

### FLoRA — Fused Forward-Backward Adapters
- **Data:** Outubro 2025
- **Fonte:** arXiv
- **Resumo:** Integra adapters diretamente nas camadas de projeção, halving GPU kernel launches. Reduz latência de inferência em 20-30%.
- **Relevância:** Nosso "adapter" é o tensor delta aplicado sobre o codebook. A ideia de fusão é análoga à nossa aplicação XOR inline.
- **Aplicação:** Validar que deltas inline (sem camada extra) mantêm performance.

### aLoRA — Activated LoRA (IBM)
- **Data:** Abril 2025
- **Fonte:** IBM Research
- **Resumo:** LoRAs "ativados" que funcionam independentemente do base model durante inferência, reciclando computações do KV cache. Speedup de 20-30x por tarefa.
- **Relevância:** Inspira nosso Neurônio Semi-Fixo: o delta pode operar com autonomia parcial sobre o cache.
- **Aplicação:** Implementar "delta ativado" que não precise ler todo o brain.crom.

### LoRA-Dash (ICLR 2025)
- **Data:** 2025
- **Fonte:** Harvard / ICLR
- **Resumo:** Framework que define "task-specific directions" (TSDs) para maximizar eficácia do fine-tuning.
- **Relevância:** TSDs podem ser mapeados como "direções" no espaço do codebook DNA.
- **Aplicação:** Usar TSDs para guiar a geração de deltas mais expressivos.

### KRAdapter — Khatri-Rao Product Adapters
- **Data:** Outubro 2025
- **Fonte:** CVPR
- **Resumo:** Usa Khatri-Rao product para gerar matrices de rank efetivo maior que LoRA standard.
- **Relevância:** Alternativa à composição linear de deltas. Multiplicação Khatri-Rao pode gerar deltas mais expressivos.
- **Aplicação:** Testar Khatri-Rao como operação de composição multi-delta (Vertente 3).

---

## 3. Deduplicação & Delta Compression de Modelos

### ZipLLM + BitX — Model-Aware Delta Compression
- **Data:** 2025 (arXiv:2505.06252)
- **Fonte:** arXiv
- **Resumo:** Trata deduplicação e compressão como pipeline unificado. BitX codifica deltas XOR entre modelos fine-tuned da mesma família. Economia de 54% no armazenamento.
- **Relevância:** **Paper mais diretamente relevante.** BitX é essencialmente nosso XOR Delta sobre Codebook, validado em escala.
- **Aplicação:** Implementar BitX como formato alternativo de delta. Comparar com nosso XOR nativo.

### D²-MoE — Delta Decomposition for Mixture-of-Experts
- **Data:** 2025 (ICML 2025, arXiv:2502.17298)
- **Fonte:** arXiv / ICML
- **Resumo:** Decompõe experts em base compartilhada + deltas únicos via Fisher info matrix + SVD. Alta compressão sem retreinar.
- **Relevância:** Multi-Brain Engine pode usar decomposição similar: base compartilhada entre neurônios + delta por neurônio.
- **Aplicação:** Implementar Fisher-based merge para Multi-Brain routing (Fase 3).

---

## 4. Inferência via SSD & Offloading

### FlexInfer (MLSys 2025)
- **Data:** 2025
- **Fonte:** MLSys
- **Resumo:** Framework com performance estimator para selecionar políticas de offloading otimizadas por fase (prefill/decode).
- **Relevância:** FUSE mount do Crompressor já faz offloading nativo. FlexInfer adiciona inteligência de política de acesso.
- **Aplicação:** Implementar estimator de acesso para otimizar quais chunks cachear em RAM.

### HILOS — Near-Storage Processing for LLMs
- **Data:** Fevereiro 2026
- **Fonte:** arXiv
- **Resumo:** Offloads attention operations para aceleradores near-storage. Throughput 7.86x maior.
- **Relevância:** O .crom no SSD já é "near-storage". HILOS valida que computação perto do storage é eficiente.
- **Aplicação:** Mover XOR Delta para executar diretamente durante FUSE read (near-storage).

### Apple LLM in Flash
- **Data:** 2024-2025
- **Fonte:** Apple Research
- **Resumo:** Windowing + row-column bundling para leitura eficiente de flash. Reutiliza neurônios ativados.
- **Relevância:** Técnicas de acesso sequencial otimizado para SSD aplicáveis ao FUSE mount.
- **Aplicação:** Implementar windowing para chunks CDC mais eficiente.

---

## 5. Mixture of Frozen Experts & Multi-Model Routing

### MoFE — Mixture of Frozen Experts
- **Data:** 2025
- **Fonte:** ACL Anthology / arXiv
- **Resumo:** Congela FFN layers dos experts e usa routing entre eles. Reduz parâmetros treináveis drasticamente.
- **Relevância:** **Análogo direto do Multi-Brain Engine.** brain.crom = frozen expert. Router = HNSW.
- **Aplicação:** Base teórica para nosso Multi-Brain routing (Fase 3).

### Brainstacks — Modular Continual Multi-Domain Fine-Tuning
- **Data:** 2026
- **Fonte:** arXiv
- **Resumo:** Adapter stacks congelados com MoE-LoRA routing e meta-router sigmoid. Composição aditiva cross-domain.
- **Relevância:** Stacks = neurônios .crom. Meta-router = nosso HNSW routing. Composição aditiva = soma ponderada de deltas.
- **Aplicação:** Implementar sigmoid meta-router para seleção de neurônios.

---

## 6. Termodinâmica & Entropia em Redes Neurais

### Shannon Entropy = Thermodynamic Entropy (arXiv:2512.02221)
- **Data:** Dezembro 2025
- **Fonte:** arXiv
- **Resumo:** Demonstra equivalência quantitativa entre Shannon entropy e Boltzmann-Gibbs entropy em sistemas de matéria condensada.
- **Relevância:** Suporta nossa medição de entropia do brain.crom e dos deltas como grandeza termodinâmica real.
- **Aplicação:** Usar framework de equivalência para interpretar drift entrópico do neurônio semi-fixo.

### Neural Thermodynamic Laws for LLMs (arXiv:2505.10571)
- **Data:** Maio 2025
- **Fonte:** arXiv
- **Resumo:** Aplica leis termodinâmicas ao treinamento de Large Language Models.
- **Relevância:** Base teórica para interpretar "temperatura" do neurônio e "energia livre" do delta.
- **Aplicação:** Modelar custo termodinâmico da aplicação de deltas como "trabalho" sobre o sistema.

### ICML 2026 Workshop: Neural Compression
- **Data:** 2026
- **Fonte:** ICML
- **Resumo:** Workshop sobre papel das medidas de informação no desenvolvimento de modelos eficientes.
- **Relevância:** Forum para apresentar results do crompressor-neuronio.
- **Aplicação:** Submissão de paper/poster sobre DNA compression + tensor delta.

---

## 7. Checkpointing & Versionamento de Modelos

### Prediction-Based Checkpoint Compression (arXiv:2506.12000)
- **Data:** 2025
- **Fonte:** arXiv
- **Resumo:** Usa checkpoints anteriores como contexto para arithmetic coding dos checkpoints atuais. Near-lossless.
- **Relevância:** Neurônio Semi-Fixo gera "versões" via delta. Técnica de predição pode comprimir log de deltas.
- **Aplicação:** Comprimir histórico de deltas do neurônio dinâmico.

---

## Tabela Resumo de Relevância

| Paper | Vertente | Prioridade | Aplicação Principal |
|:---|:---|:---|:---|
| ZipLLM BitX | Fixo | 🔴 Alta | Delta XOR format |
| LLVQ | Todas | 🔴 Alta | VQ sem codebook gigante |
| MoFE | Dinâmico | 🔴 Alta | Multi-Brain teórica |
| TurboQuant | Fixo/Semi | 🟡 Média | Cache compression |
| aLoRA | Semi-Fixo | 🟡 Média | Delta ativado |
| Brainstacks | Dinâmico | 🟡 Média | Meta-router |
| arXiv:2512.02221 | Todas | 🟡 Média | Entropy framework |
| FlexInfer | Todas | 🟢 Baixa | Policy estimator |
| D²-MoE | Dinâmico | 🟢 Baixa | Fisher merge |
| HILOS | Fixo | 🟢 Baixa | Near-storage |

---

> Atualizar este documento conforme novos papers surgirem.
