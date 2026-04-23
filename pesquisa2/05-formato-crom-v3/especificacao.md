# Especificação do Formato .crom v3

---

## Evolução

| Versão | Conteúdo | Pesquisa |
|--------|----------|----------|
| v1 | Codebook único + índices | pesquisa0 |
| v2 | Header + codebook + índices (pesos GPT-2) | pesquisa1 |
| **v3** | **Modelo completo: config + embeddings + N×(codebook+índices) + LM head** | **pesquisa2** |

## Layout Binário

```
Offset  Size          Content
──────  ──────────    ────────────────────────
0x00    4 bytes       Magic: "CROM"
0x04    1 byte        Version: 3
0x05    4 bytes       n_layers (uint32)
0x09    4 bytes       n_heads (uint32)
0x0D    4 bytes       d_model (uint32)
0x11    4 bytes       d_ff (uint32)
0x15    4 bytes       vocab_size (uint32)
0x19    4 bytes       max_seq_len (uint32)
0x1D    4 bytes       K (uint32) — codebook size
0x21    4 bytes       D (uint32) — centroid dim
0x25    4 bytes       total_crom_layers (uint32)

─── SECTION: EMBEDDINGS ───
0x29    vocab×dim×2   Token Embedding (Float16)
...     max_len×dim×2 Position Embedding (Float16)

─── SECTION: LAYERS (repeated n_layers times) ───
...     K×D×4         Codebook Attention (Float32) — Q,K,V,O compartilham
...     n_blocks×2    Indices Attention (uint16)
...     K×D×4         Codebook FFN (Float32) — up,down compartilham
...     n_blocks×2    Indices FFN (uint16)
...     dim×4         LayerNorm1 weight (Float32)
...     dim×4         LayerNorm1 bias (Float32)
...     dim×4         LayerNorm2 weight (Float32)
...     dim×4         LayerNorm2 bias (Float32)

─── SECTION: HEAD ───
...     dim×4         Final LayerNorm weight
...     dim×4         Final LayerNorm bias
...     vocab×dim×2   LM Head (Float16 ou tied com embedding)

─── SECTION: METADATA ───
...     variable      JSON metadata (training config, metrics, etc.)
```

## Validação

```
1. save_cromgpt(model, "model.crom")
2. model2 = load_cromgpt("model.crom")
3. output1 = model(test_input)
4. output2 = model2(test_input)
5. assert torch.allclose(output1, output2)  # MUST PASS
```

## Comparação de Tamanho Esperado

| Formato | Tamanho Estimado (125M) |
|---------|------------------------|
| PyTorch .pt (Float32) | ~500 MB |
| SafeTensors (Float16) | ~250 MB |
| GGUF (Q4) | ~70 MB |
| **.crom v3** | **~50-80 MB** |
