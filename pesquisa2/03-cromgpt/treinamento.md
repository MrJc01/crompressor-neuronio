# Treinamento CromGPT — Loop e Hiperparâmetros

---

## Hyperparams

| Parâmetro | Valor | Referência |
|-----------|-------|------------|
| Optimizer | AdamW | Padrão para Transformers |
| Learning Rate | 3e-4 | GPT-2 original |
| LR Codebook | 1e-3 | Maior que pesos normais (codebook precisa convergir rápido) |
| Weight Decay | 0.1 | Regularização padrão |
| Warmup | 2000 steps | Linear warmup |
| Scheduler | Cosine decay | Reduz LR gradualmente |
| Batch Size | 4 | Limitado pela VRAM T4 |
| Gradient Accumulation | 8 | Batch efetivo = 32 |
| Gradient Clipping | max_norm=1.0 | Evita explosão de gradientes |
| Max Seq Len | 512 | Limitado pela VRAM |
| Epochs | 1-3 | Limitado pelo Colab |

---

## Loop de Treinamento

```python
for epoch in range(n_epochs):
    for batch in dataloader:
        # Forward
        logits = model(batch.input_ids)
        loss = cross_entropy(logits, batch.labels)
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Step
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        
        # Periodicamente: re-assign indices (nearest neighbor)
        if step % reassign_every == 0:
            reassign_codebook_indices(model)
        
        # Logging
        if step % log_every == 0:
            log(step, loss, perplexity, lr, codebook_utilization)
        
        # Checkpoint
        if step % save_every == 0:
            save_checkpoint(model, optimizer, step)
```

---

## Re-Assignment de Índices

A cada N steps, recalculamos quais centróides cada bloco de pesos usa:

```python
def reassign_codebook_indices(layer):
    """Para cada bloco do peso, acha o centróide mais próximo."""
    with torch.no_grad():
        for block_idx in range(n_blocks):
            block = continuous_weights[block_idx]  # [D]
            distances = torch.cdist(block.unsqueeze(0), layer.codebook)  # [1, K]
            layer.indices[block_idx] = distances.argmin()
```

---

## Monitoramento de Codebook

```python
def codebook_utilization(model):
    """% de centróides que estão sendo usados."""
    for layer in model.crom_layers():
        used = len(torch.unique(layer.indices))
        total = layer.codebook.shape[0]  # K
        util = used / total * 100
        if util < 50:
            log.warning(f"⚠️ Codebook utilização baixa: {util:.0f}%")
```

---

## Checkpointing (Colab)

```python
def save_checkpoint(model, optimizer, step, path="checkpoint.pt"):
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'codebook_stats': get_codebook_stats(model),
    }, path)
    # Upload para Google Drive como backup
    shutil.copy(path, "/content/drive/MyDrive/cromgpt/")
```

---

## Métricas por Step

| Métrica | O que significa |
|---------|----------------|
| `loss` | CrossEntropy — deve diminuir |
| `ppl` | Perplexidade = exp(loss) — deve diminuir |
| `lr` | Learning rate atual |
| `cb_util` | % centróides usados — deve ser >50% |
| `grad_norm` | Norma dos gradientes — não deve explodir |
| `tokens/s` | Throughput — para estimar tempo restante |
