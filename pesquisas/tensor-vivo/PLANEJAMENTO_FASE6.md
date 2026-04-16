# 🧬 Tensor-Vivo — Planejamento Fase 6+

> **Atualizado:** 2026-04-16
> **Status:** Exp5 v2 concluído — Tese validada em MLP, CNN, Transformer
> **Saldo Vast.ai:** ~$5.50

---

## Situação Atual

### Resultados Consolidados
```
  Arquitetura     | Baseline | Codebook | Gap    | Compressão | Recovery
  ----------------+----------+----------+--------+------------+---------
  MNIST MLP       |  97.53%  |  97.56%  | +0.03% |    40.8x   |  100.0%
  CIFAR-10 CNN    |  77.86%  |  77.66%  | -0.20% |   145.3x   |   99.7%
  GPT-2 Transf.   |  91.51%  |  83.72%  | -7.80% |   389.5x   |   91.5%
```

### Problemas Abertos
1. **Gap de 8% no Transformer** — K=256 insuficiente para 124M params
2. **Comparação LoRA pendente** — v1 tinha LoRA mas zero codebook layers
3. **Formato .crom não definido** — precisa spec binária para codebooks
4. **Integração Go inexistente** — CodebookLinear só existe em Python

---

## Fase 6: Fechar o Gap do Transformer

### Objetivo
Reduzir gap de 8% → <2% no GPT-2 SST-2, mantendo compressão >100x.

### Estratégias (ordem de prioridade)

#### 6A. Aumentar K e epochs (mais barato, mais provável)
- K=512 B=16, K=1024 B=16, K=2048 B=16
- 10 epochs em vez de 5
- Learning rate sweep: 1e-4, 5e-4, 1e-3
- **Custo estimado:** ~$1.00 no Vast.ai (RTX A4000, 2-3h)

#### 6B. Codebook por camada (adaptativo)
- Calcular variância de pesos por camada
- Alocar K proporcionalmente: camadas com mais variância → mais K
- Exemplo: c_attn (768→2304) precisa de mais K que c_proj (768→768)

#### 6C. Residual Codebook (duas passadas)
- 1ª quant: K=256 → treinar codebook
- 2ª quant: quantizar os RESÍDUOS com segundo codebook K=128
- Total: 2 codebooks menores > 1 codebook grande

#### 6D. Mixed Precision
- Manter embeddings (wte, wpe) e LayerNorm em full precision
- Só quantizar attention+MLP projections
- Pode melhorar accuracy sem custo de compressão

---

## Fase 7: Comparação Formal com LoRA

### Objetivo
Provar que Codebook Learning é competitivo com LoRA no mesmo budget de params.

### Experimento
- Codebook K=256 B=16: 319K params → accuracy X%
- LoRA rank=2: ~300K params → accuracy Y%
- LoRA rank=4: ~600K params → accuracy Z%
- Codebook K=512 B=32: 909K params → accuracy W%
- LoRA rank=8: ~1.2M params → accuracy V%

### Métricas
- Accuracy por param treinável
- Tempo de convergência
- Overhead de memória (codebook indices vs LoRA low-rank matrices)

---

## Fase 8: Implementação Go + Formato .crom

### Objetivo
Integrar codebook learning no motor Go do Crompressor.

### 8A. Formato binário .crom v2
```
[Header 64 bytes]
  magic: "CROM" (4 bytes)
  version: uint16
  model_name: string (32 bytes)
  num_layers: uint16
  total_codebook_params: uint32
  total_indices: uint32
  checksum: sha256 (32 bytes)

[Per-layer entry]
  layer_name: string (64 bytes)
  K: uint16
  block_size: uint16
  shape: [2]uint32
  codebook: float32[K × block_size]
  indices: uint16[num_blocks]  // uint16 se K ≤ 65536

[Footer]
  merkle_root: sha256 (32 bytes)
```

### 8B. CodebookLinear em Go
- Carregar .crom
- Reconstruir peso: `W = codebook[indices].reshape(shape)`
- Forward pass: `output = input @ W + bias`
- Integrar com FUSE driver para servir modelos

### 8C. FUSE Driver para Modelos Codebook
- Interceptar leitura de arquivos .gguf / .safetensors
- Servir pesos reconstruídos do codebook em tempo real
- Zero cópia: reconstruir on-the-fly

---

## Fase 9: Paper "Codebook-as-LoRA"

### Estrutura
1. **Abstract** — Codebook quantization + learning como PEFT alternativo
2. **Introduction** — VQ meets parameter-efficient fine-tuning
3. **Method** — CodebookLinear, frozen indices, trainable centroids
4. **Experiments** — MLP (MNIST), CNN (CIFAR-10), Transformer (GPT-2, SST-2)
5. **Results** — Accuracy-compression Pareto frontier vs LoRA
6. **Analysis** — Regularization effect, per-layer K allocation, convergence
7. **Discussion** — Limitations (gap scaling), future work (residual codebook)
8. **Conclusion**

### Venues
- ICLR 2027 (deadline ~Oct 2026)
- NeurIPS 2027 (deadline ~May 2027)
- EMNLP 2026 (deadline ~Jun 2026) — se correr!

---

## Checklist de Tarefas

### Fase 6: Fechar Gap Transformer
- [ ] Criar `run_exp5_v3_sweep.py` com grid de K e epochs
  - [ ] K=512 B=16, K=1024 B=16, K=2048 B=16
  - [ ] 10 epochs para cada config
  - [ ] LR sweep: 1e-4, 5e-4, 1e-3
  - [ ] Reutilizar baseline salvo (não retreinar)
  - [ ] Salvar checkpoint por epoch (early stopping analysis)
- [ ] Provisionar RTX A4000 ou 4090 no Vast.ai (~$1.00)
- [ ] Rodar sweep via SSH (nohup + monitoramento)
- [ ] Analisar resultados: accuracy vs K, accuracy vs epoch, accuracy vs LR
- [ ] Identificar melhor config para gap <2%
- [ ] Se gap persistir: implementar Residual Codebook (6C)
- [ ] Se gap persistir: implementar per-layer K adaptativo (6B)
- [ ] Atualizar CONCLUSOES.md com resultados v3

### Fase 7: Comparação LoRA
- [ ] Adicionar LoRA ao script v3 (mesmo SST-2, mesmo baseline)
- [ ] Testar LoRA rank=2, 4, 8, 16
- [ ] Calcular params treináveis para cada rank
- [ ] Gerar tabela comparativa: Codebook vs LoRA (accuracy, params, tempo)
- [ ] Gerar gráfico accuracy vs params (Pareto frontier)
- [ ] Documentar vantagens/desvantagens de cada abordagem

### Fase 8: Integração Go + .crom
- [ ] Definir spec binária do formato .crom v2
- [ ] Implementar writer em Python (exportar codebook → .crom)
- [ ] Implementar reader em Go (carregar .crom → reconstruir peso)
- [ ] Implementar CodebookLinear.Forward() em Go
- [ ] Benchmark: latência de reconstrução de peso
- [ ] Integrar com FUSE driver existente
- [ ] Teste end-to-end: Python treina → exporta .crom → Go carrega → infere

### Fase 9: Paper
- [ ] Outline do paper (structure acima)
- [ ] Gerar todas as tabelas de resultados
- [ ] Gerar figuras: convergence curves, Pareto frontier, per-layer analysis
- [ ] Escrever seção Method
- [ ] Escrever seção Experiments
- [ ] Escrever Results + Analysis
- [ ] Revisar com feedback externo
- [ ] Submeter

### Infraestrutura
- [ ] Regenerar API key do Vast.ai (a atual foi exposta neste chat)
- [ ] Documentar workflow Vast.ai (CLI commands, SSH setup)
- [ ] Criar script `vastai_run.sh` genérico para provisionar + rodar + destruir
- [ ] Git push de cada fase
- [ ] Manter CONCLUSOES.md atualizado a cada resultado

---

## Prioridades

| Prioridade | Tarefa | Custo | Impacto |
|---|---|---|---|
| 🔴 P0 | Regenerar API key Vast.ai | 0 | Segurança |
| 🔴 P0 | Sweep K+epochs (Fase 6A) | ~$1.00 | Fechar gap |
| 🟡 P1 | Comparação LoRA formal (Fase 7) | ~$0.50 | Validação |
| 🟡 P1 | Formato .crom v2 spec | 0 | Fundação |
| 🟢 P2 | CodebookLinear em Go | 0 | Integração |
| 🟢 P2 | Paper outline | 0 | Publicação |
| ⚪ P3 | Residual Codebook | ~$0.50 | Pesquisa |
| ⚪ P3 | FUSE driver codebook | 0 | Produto |

---

## Estimativa de Custos (Vast.ai)

| Fase | GPU | Tempo | Custo |
|---|---|---|---|
| 6A: K sweep | RTX A4000 $0.08/hr | ~3-4h | ~$0.30 |
| 6A: Epoch sweep | RTX A4000 $0.08/hr | ~4-5h | ~$0.40 |
| 7: LoRA comparison | RTX A4000 $0.08/hr | ~3-4h | ~$0.30 |
| 6C: Residual (se necessário) | RTX A4000 $0.08/hr | ~3h | ~$0.25 |
| **Total estimado** | | | **~$1.25** |

**Saldo restante após tudo: ~$4.25** — margem confortável para erros e retentativas.
