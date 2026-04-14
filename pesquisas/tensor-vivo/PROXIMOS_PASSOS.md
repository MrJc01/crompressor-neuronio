# 🗺️ Próximos Passos — Tensor-Vivo Checklist Extenso

> **Pré-requisito:** Fases 0-2 completas. Tese central validada em MNIST MLP.
> **Objetivo:** Escalar, aprofundar e integrar os resultados.

---

## Fase 3: Escalar para CIFAR-10 CNN (~4-6h)

> **Por quê:** MNIST MLP é trivial. Precisamos provar que funciona com
> convoluções, múltiplas camadas, e dados visuais reais.

### Setup
- [ ] Criar `exp3_cifar_cnn/`
- [ ] Definir CNN: Conv2d(3→32→64) + MaxPool + Linear(64*8*8→256→10)
- [ ] Treinar até accuracy > 85% no CIFAR-10
- [ ] Salvar modelo treinado

### Codebook Quantization em CNN
- [ ] Adaptar `codebook_quantize.py` para Conv2d layers
- [ ] Decidir granularidade: por filtro? por canal? por kernel?
  - [ ] **Opção A:** Cada filtro (out_ch × in_ch × 3 × 3) como 1 bloco
  - [ ] **Opção B:** Cada kernel (in_ch × 3 × 3) como 1 bloco
  - [ ] **Opção C:** Flatten e chunkar por block_size fixo (como no MLP)
- [ ] Testar K = {32, 64, 128, 256, 512} para cada opção
- [ ] Medir accuracy × compressão para cada combinação
- [ ] Salvar em `dados/exp3_quantize_results.json`

### CodebookConv2d
- [ ] Implementar `CodebookConv2d(nn.Module)`
  - [ ] Pesos reconstruídos via `codebook[indices].reshape(out_ch, in_ch, kH, kW)`
  - [ ] Forward: `F.conv2d(x, reconstructed_weight, bias, stride, padding)`
  - [ ] Codebook treinável, indices congelados
- [ ] Criar `CodebookCNN` substituindo todas as Conv2d
- [ ] Training loop: 20 epochs, medir recovery
- [ ] Comparar pré-treino vs pós-treino vs baseline
- [ ] Salvar em `dados/exp3_learning_results.json`

### Análise
- [ ] `resultados.md` com:
  - [ ] Tabela por tipo de camada (Conv vs Linear): qual comprime melhor?
  - [ ] Curva accuracy × K × tipo de camada
  - [ ] Conclusão: CNN funciona tão bem quanto MLP?

---

## Fase 4: CDC com LSH — Dedup por Similaridade (~4-6h)

> **Por quê:** O Exp0 mostrou que CDC hash exato dá 0% dedup.
> Mas se usarmos Locality-Sensitive Hashing em vez de SHA-256,
> chunks "quase iguais" podem colidir e gerar dedup real.

### Implementação
- [ ] Criar `exp4_lsh_dedup/`
- [ ] Implementar `lsh_cdc.py`:
  - [ ] Random projection LSH: `hash(W) = sign(R @ W)` onde R é random matrix
  - [ ] Extrair pesos do MLP e do CNN
  - [ ] Para cada bloco de pesos, gerar LSH hash
  - [ ] Contar colisões LSH (blocos com hash igual)
  - [ ] Comparar com hash exato (SHA-256)
- [ ] Variar número de hiperplanos LSH: {4, 8, 16, 32}
- [ ] Medir:
  - [ ] Taxa de "dedup LSH" por camada
  - [ ] Falsos positivos: blocos com mesmo hash LSH mas distância euclidiana > threshold
  - [ ] Falsos negativos: blocos próximos que não colidiram

### Dedup LSH → Weight Sharing
- [ ] Para blocos com mesmo hash LSH, substituir pelo centróide
- [ ] Medir accuracy com weight sharing via LSH
- [ ] Comparar com K-Means (que é mais sofisticado)
- [ ] Salvar em `dados/exp4_results.json`

### Análise
- [ ] Se LSH dedup > 0%: CDC com LSH é uma forma válida de detecção de redundância
- [ ] Se LSH dedup ≈ K-Means dedup: LSH é mais barato computacionalmente
- [ ] `resultados.md` com gráficos e conclusões

---

## Fase 5: Escalar para Transformer (~6-8h)

> **Por quê:** O teste definitivo. Se funcionar em GPT-2 small (124M params),
> a tese tem escala real.

### Setup
- [ ] Criar `exp5_transformer/`
- [ ] Usar GPT-2 small pré-treinado (HuggingFace)
- [ ] Definir task: classificação de sentimento (SST-2) ou text completion
- [ ] Medir baseline (accuracy ou perplexity)

### Codebook Quantization em Transformer
- [ ] Analisar estrutura dos pesos:
  - [ ] Q, K, V projections (768 × 768)
  - [ ] MLP up/down projections (768 × 3072)
  - [ ] Layer norms, embeddings
- [ ] Decidir quais camadas quantizar (todas? só MLP? só QKV?)
- [ ] Testar K = {256, 512, 1024, 2048, 4096}
- [ ] Medir perplexity × compressão
- [ ] Salvar em `dados/exp5_quantize_results.json`

### Codebook Learning no Transformer
- [ ] Implementar `CodebookAttention` e `CodebookMLP` para blocos Transformer
- [ ] Congelar indices, treinar codebook
- [ ] Fine-tune: 3-5 epochs no SST-2
- [ ] Medir recovery de perplexity/accuracy
- [ ] Comparar com LoRA rank=4 (mesmo nº de params treináveis)

### Análise
- [ ] Se recovery > 90%: tese confirmada em escala real
- [ ] Se codebook < LoRA: investigar por quê
- [ ] Se codebook > LoRA: paper material
- [ ] `resultados.md` com tabelas comparativas

---

## Fase 6: Integração com Motor Go .crom (~4-6h)

> **Por quê:** A pesquisa Python provou o conceito. Agora integrar
> com o ecossistema Go do Crompressor para uso real.

### Formato .crom para Codebooks
- [ ] Definir extensão do header .crom para codebooks:
  ```
  [64B Standard Header]
  [4B flag: CODEBOOK_QUANTIZED]
  [4B K: número de centróides]
  [4B block_size]
  [4B num_layers]
  [N × LayerMeta: name_len, name, shape, indices_offset, codebook_offset]
  [Codebook data: K × block_size × float32]
  [Indices data: num_blocks × uint16]
  ```
- [ ] Implementar writer em Go: `pkg/crom/codebook_writer.go`
- [ ] Implementar reader em Go: `pkg/crom/codebook_reader.go`
- [ ] Teste: Python gera codebook → Go lê → bytes idênticos

### Exportar de Python para .crom
- [ ] Script `export_codebook.py`:
  - [ ] Carregar CodebookMLP treinado
  - [ ] Serializar codebook + indices em formato binário
  - [ ] Escrever .crom com header correto
- [ ] Verificar que Go lê o .crom e reconstrói os pesos corretamente

### FUSE com Codebook
- [ ] Adaptar `crom_fs.go` para servir modelos codebook-quantized
- [ ] Read() reconstrói pesos on-the-fly via codebook[indices]
- [ ] Benchmark: latência de reconstrução via FUSE vs pesos raw

---

## Fase 7: DNA Encoding dos Centróides (~3-4h)

> **Por quê:** O Crompressor usa DNA Base-4. Testar se codificar
> centróides em DNA muda algo vs float32 puro.

- [ ] Criar `exp7_dna_codebook/`
- [ ] Implementar encoding: float32 → 4-bit quantize → DNA (A,T,G,C)
- [ ] Implementar decoding: DNA → 4-bit → float32 (com perda controlada)
- [ ] Treinar codebook com centróides DNA-encoded
- [ ] Medir accuracy vs codebook float32
- [ ] Se accuracy similar: DNA encoding é viável como representação
- [ ] Se accuracy cai muito: DNA precisa de mais bits (8-bit DNA = 2 bases por byte)

---

## Fase 8: Codebook Compartilhado Cross-Model (~3-4h)

> **Por quê:** Se dois modelos compartilham centróides no codebook,
> temos "dedup de conhecimento" entre modelos — o sonho do Crompressor.

- [ ] Treinar 2 modelos diferentes no MNIST (MLP configs diferentes)
- [ ] Extrair codebooks de ambos
- [ ] Merge codebooks: K-Means sobre a UNIÃO dos centróides
- [ ] Reatribuir indices de cada modelo ao codebook merged
- [ ] Medir accuracy de cada modelo com codebook compartilhado
- [ ] Calcular "taxa de sharing": quantos centróides são usados por ambos?
- [ ] Se sharing > 50%: modelos diferentes compartilham representações
- [ ] Testar com modelos de arquiteturas diferentes (MLP vs CNN)

---

## Fase 9: Paper Draft (~2-3h)

> Se as fases 3-5 confirmarem os resultados em escala.

- [ ] Outline do paper:
  - [ ] Title: "Codebook-as-LoRA: Weight Representation via Learned Codebooks"
  - [ ] Abstract
  - [ ] Introduction (compressão = cognição, CDC background)
  - [ ] Method (CodebookLinear, training regime)
  - [ ] Experiments (MNIST, CIFAR, GPT-2)
  - [ ] Results tables
  - [ ] Comparison with LoRA
  - [ ] Discussion (CDC hash vs LSH, DNA encoding prospects)
  - [ ] Conclusion
- [ ] Figuras: curvas accuracy × K, convergence plots
- [ ] Submeter para workshop (ICML, NeurIPS, ou MLSys)

---

## Priorização Recomendada

| Prioridade | Fase | Tempo | Impacto |
|---|---|---|---|
| 🔴 Alta | **Fase 3** (CIFAR CNN) | 4-6h | Valida que funciona além de MNIST |
| 🔴 Alta | **Fase 5** (Transformer) | 6-8h | Teste definitivo de escala |
| 🟡 Média | **Fase 4** (CDC LSH) | 4-6h | Resolve o gap do CDC hash exato |
| 🟡 Média | **Fase 6** (Integração Go) | 4-6h | Conecta pesquisa ao produto |
| 🟢 Baixa | **Fase 7** (DNA Encoding) | 3-4h | Exploratório, nice-to-have |
| 🟢 Baixa | **Fase 8** (Cross-Model) | 3-4h | Exploratório avançado |
| ⚪ Condicional | **Fase 9** (Paper) | 2-3h | Só se fases 3+5 confirmarem |

**Sugestão:** Fazer Fase 3 → Fase 5 → Fase 4 nessa ordem.
