#!/usr/bin/env python3
"""
🧬 Tensor-Vivo — Exp5 v2: Codebook Learning em GPT-2 Small
Versão standalone para Vast.ai / RunPod / qualquer GPU.
Sem dependências do Google Colab.

Uso:
  pip install torch transformers datasets scikit-learn
  python run_exp5_vastai.py

Resultados salvos em ./resultados/
Tempo estimado: ~40-60 min (RTX 4090) / ~20-30 min (A100)
"""

import os, json, time, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import MiniBatchKMeans

# ════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════
RESULTS_DIR = "./resultados"
BASELINE_PATH = f"{RESULTS_DIR}/gpt2_sst2_baseline.pt"
BATCH_SIZE = 32
MAX_LEN = 128
BASELINE_EPOCHS = 3
CODEBOOK_EPOCHS = 5
CODEBOOK_CONFIGS = [
    (256, 16),   # Granular — melhor resultado esperado
    (512, 32),   # Compressão maior
]

os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")
if torch.cuda.is_available():
    print(f"    GPU: {torch.cuda.get_device_name(0)}")
    print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  Sem GPU! Vai demorar MUITO. Considere usar --cpu-ok para confirmar.")
    if "--cpu-ok" not in sys.argv:
        sys.exit(1)

# ════════════════════════════════════════════════════════════
# 1. Dataset + Tokenizer
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  FASE 1: Preparar dados e modelo baseline")
print("=" * 70)

from transformers import GPT2ForSequenceClassification, GPT2Tokenizer
from transformers.pytorch_utils import Conv1D
from datasets import load_dataset

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("glue", "sst2")

def tokenize_fn(examples):
    return tokenizer(examples["sentence"], truncation=True,
                     padding="max_length", max_length=MAX_LEN, return_tensors=None)

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["sentence", "idx"])
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch")

train_loader = torch.utils.data.DataLoader(
    tokenized["train"], batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = torch.utils.data.DataLoader(
    tokenized["validation"], batch_size=64, num_workers=2, pin_memory=True)

print(f"  Train: {len(tokenized['train']):,} amostras, {len(train_loader)} batches/epoch")
print(f"  Val:   {len(tokenized['validation']):,} amostras")

# ════════════════════════════════════════════════════════════
# 2. Evaluate function
# ════════════════════════════════════════════════════════════
def evaluate_model(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            correct += (outputs.logits.argmax(-1) == labels).sum().item()
            total += labels.size(0)
    return correct / total * 100

# ════════════════════════════════════════════════════════════
# 3. Baseline (carregar ou treinar)
# ════════════════════════════════════════════════════════════
model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
model.config.pad_token_id = tokenizer.pad_token_id

if os.path.exists(BASELINE_PATH):
    print(f"\n  📂 Carregando baseline de {BASELINE_PATH}...")
    model.load_state_dict(torch.load(BASELINE_PATH, weights_only=True, map_location="cpu"))
    model = model.to(device)
    baseline_acc = evaluate_model(model, val_loader)
    print(f"  ✅ Baseline: {baseline_acc:.2f}%")
else:
    print(f"\n  🔧 Treinando baseline ({BASELINE_EPOCHS} epochs)...")
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, BASELINE_EPOCHS + 1):
        model.train(); total_loss = 0; t0 = time.time()
        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(input_ids=input_ids, attention_mask=attention_mask).logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            if (i + 1) % 500 == 0:
                print(f"      step {i+1}/{len(train_loader)} loss={total_loss/(i+1):.4f}")
        acc = evaluate_model(model, val_loader)
        print(f"    Epoch {epoch}: loss={total_loss/len(train_loader):.4f} acc={acc:.2f}% ({time.time()-t0:.0f}s)")
    torch.save(model.state_dict(), BASELINE_PATH)
    baseline_acc = evaluate_model(model, val_loader)
    print(f"  ✅ Baseline salvo: {baseline_acc:.2f}%")

total_params = sum(p.numel() for p in model.parameters())
target_params = sum(p.numel() for n, p in model.named_parameters()
                    if any(n.endswith(s) for s in [
                        "attn.c_attn.weight", "attn.c_proj.weight",
                        "mlp.c_fc.weight", "mlp.c_proj.weight"]))
print(f"  Total params:  {total_params:,}")
print(f"  Target params: {target_params:,} ({target_params/total_params*100:.1f}%)")

# Verificar que são Conv1D
for name, module in model.named_modules():
    if "transformer.h.0.attn.c_attn" == name:
        print(f"  Tipo da camada: {type(module).__name__} (weight: {module.weight.shape})")
        break

del model; torch.cuda.empty_cache()

# ════════════════════════════════════════════════════════════
# 4. CodebookConv1D (CORRIGIDO para GPT-2)
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  FASE 2: Codebook Learning (Conv1D corrigido)")
print("=" * 70)

class CodebookConv1D(nn.Module):
    """Conv1D do GPT-2 com pesos reconstruídos via codebook treinável.

    Conv1D (HuggingFace): weight=(in_f, out_f), forward=x @ W + b
    Diferente de nn.Linear que faz x @ W.T + b
    """
    def __init__(self, original, K, block_size):
        super().__init__()
        assert isinstance(original, Conv1D)
        self.nf = original.nf
        self.weight_shape = original.weight.shape
        self.block_size = block_size

        w = original.weight.detach().cpu().numpy().flatten()
        n = len(w); self.original_numel = n
        if n % block_size:
            w = np.concatenate([w, np.zeros(block_size - n % block_size)])

        blocks = w.reshape(-1, block_size)
        actual_K = min(K, blocks.shape[0])
        km = MiniBatchKMeans(n_clusters=actual_K, random_state=42,
                             batch_size=min(1000, blocks.shape[0]))
        km.fit(blocks)

        self.codebook = nn.Parameter(torch.tensor(km.cluster_centers_, dtype=torch.float32))
        self.register_buffer('indices', torch.tensor(km.labels_, dtype=torch.long))
        self.bias = nn.Parameter(original.bias.detach().clone()) if original.bias is not None else None
        self.actual_K = actual_K
        self.num_blocks = blocks.shape[0]

    def forward(self, x):
        W = self.codebook[self.indices].reshape(-1)[:self.original_numel]
        W = W.reshape(self.weight_shape)
        out = torch.matmul(x, W)
        if self.bias is not None:
            out = out + self.bias
        return out


def replace_conv1d_with_codebook(model, K, block_size):
    """Substitui Conv1D attention/MLP por CodebookConv1D."""
    replacements = {}
    for name, module in model.named_modules():
        if isinstance(module, Conv1D) and any(t in name for t in ["c_attn","c_proj","c_fc"]):
            replacements[name] = module

    for name, original in replacements.items():
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        setattr(parent, parts[-1], CodebookConv1D(original, K, block_size))

    return model, len(replacements)


# ════════════════════════════════════════════════════════════
# 5. Training loop
# ════════════════════════════════════════════════════════════
learning_results = {
    "baseline_accuracy": baseline_acc,
    "total_params": total_params,
    "target_params": target_params,
    "model": "GPT2ForSequenceClassification",
    "task": "SST-2",
    "version": "v2-conv1d-fixed",
    "experiments": [],
}

for K, block_size in CODEBOOK_CONFIGS:
    print(f"\n{'━'*70}")
    print(f"  K={K}, Block={block_size}")
    print(f"{'━'*70}")

    # Fresh model → load baseline → replace Conv1D
    mdl = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
    mdl.config.pad_token_id = tokenizer.pad_token_id
    mdl.load_state_dict(torch.load(BASELINE_PATH, weights_only=True, map_location="cpu"))
    mdl, n_replaced = replace_conv1d_with_codebook(mdl, K, block_size)
    mdl = mdl.to(device)

    # Freeze all → unfreeze codebook + bias + LN + classifier
    for p in mdl.parameters(): p.requires_grad = False
    for name, p in mdl.named_parameters():
        if any(k in name for k in ["codebook", "bias", "ln_", "ln_f", "score"]):
            p.requires_grad = True

    pre_acc = evaluate_model(mdl, val_loader)
    trainable = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    compression = total_params / trainable if trainable else 0

    # Codebook info
    cb_info = []
    total_cb_params = 0
    for name, mod in mdl.named_modules():
        if isinstance(mod, CodebookConv1D):
            cp = mod.actual_K * mod.block_size
            total_cb_params += cp
            cb_info.append({"name": name, "K": mod.actual_K, "blocks": mod.num_blocks,
                            "block_size": mod.block_size, "cb_params": cp,
                            "shape": list(mod.weight_shape)})

    print(f"  Layers replaced: {n_replaced}")
    print(f"  Codebook params: {total_cb_params:,}")
    print(f"  Total trainable: {trainable:,} ({compression:.1f}x compressão)")
    print(f"  Pré-treino: {pre_acc:.2f}%")
    for li in cb_info[:3]:
        print(f"    {li['name']:<40} K={li['K']:>5} blocks={li['blocks']:>7} cb={li['cb_params']:>8,}")
    if len(cb_info) > 3:
        print(f"    ... +{len(cb_info)-3} camadas")

    print(f"\n  {'Ep':>4} {'Acc%':>8} {'Loss':>10} {'Δ Pre':>8} {'Δ Base':>8} {'Time':>6}")
    print("  " + "-" * 50)

    optimizer = optim.AdamW([p for p in mdl.parameters() if p.requires_grad],
                            lr=5e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CODEBOOK_EPOCHS)
    criterion = nn.CrossEntropyLoss()
    epochs_data = []

    for epoch in range(1, CODEBOOK_EPOCHS + 1):
        mdl.train(); total_loss = 0; t0 = time.time()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            loss = criterion(mdl(input_ids=input_ids, attention_mask=attention_mask).logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        acc = evaluate_model(mdl, val_loader)
        ep_time = time.time() - t0
        dp = acc - pre_acc; db = acc - baseline_acc
        print(f"  {epoch:>4} {acc:>8.2f} {avg_loss:>10.4f} {dp:>+7.2f}% {db:>+7.2f}% {ep_time:>5.0f}s")
        epochs_data.append({"epoch": epoch, "accuracy": round(acc, 4),
                            "loss": round(avg_loss, 6),
                            "delta_pre": round(dp, 4), "delta_base": round(db, 4),
                            "epoch_seconds": round(ep_time, 1)})

    final_acc = epochs_data[-1]["accuracy"]
    best_acc = max(e["accuracy"] for e in epochs_data)
    total_time = sum(e["epoch_seconds"] for e in epochs_data)
    print(f"\n  📊 pré={pre_acc:.2f}% → pós={final_acc:.2f}% (best={best_acc:.2f}%)")
    print(f"     Params: {trainable:,} ({compression:.1f}x) | Tempo: {total_time:.0f}s")

    learning_results["experiments"].append({
        "K": K, "block_size": block_size,
        "pre_train_accuracy": round(pre_acc, 4),
        "post_train_accuracy": final_acc,
        "best_epoch_accuracy": best_acc,
        "recovery": round(final_acc - pre_acc, 4),
        "gap_to_baseline": round(baseline_acc - final_acc, 4),
        "trainable_params": trainable,
        "param_compression": round(compression, 2),
        "codebook_params": total_cb_params,
        "layers_replaced": n_replaced,
        "training_time_seconds": round(total_time, 1),
        "layer_info": cb_info[:4],
        "epochs": epochs_data,
    })
    del mdl; torch.cuda.empty_cache()

# ════════════════════════════════════════════════════════════
# 6. Salvar resultados
# ════════════════════════════════════════════════════════════
out_path = f"{RESULTS_DIR}/exp5_learning_results_v2.json"
with open(out_path, "w") as f:
    json.dump(learning_results, f, indent=2)

# ════════════════════════════════════════════════════════════
# 7. Veredicto
# ════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  🏆 VEREDICTO — CODEBOOK LEARNING EM GPT-2 TRANSFORMER")
print(f"{'='*70}")

best = max(learning_results["experiments"], key=lambda x: x["post_train_accuracy"])

print(f"\n  Baseline:    {baseline_acc:.2f}% ({total_params:,} params)")
print(f"  Codebook:    {best['post_train_accuracy']:.2f}% ({best['trainable_params']:,} params)")
print(f"  Compressão:  {best['param_compression']:.1f}x")
print(f"  Gap:         {best['gap_to_baseline']:+.2f}%")
print(f"  Layers:      {best['layers_replaced']} Conv1D substituídas")
print(f"  CB params:   {best['codebook_params']:,}")

print(f"\n  📊 Evolução da Tese:")
print(f"    {'Exp':<20} {'Base':>7} {'CB':>7} {'Gap':>7} {'Comp':>8}")
print("    " + "-" * 52)
print(f"    {'MNIST MLP':<20} {'97.53%':>7} {'97.56%':>7} {'+0.03%':>7} {'40.8x':>8}")
print(f"    {'CIFAR-10 CNN':<20} {'77.86%':>7} {'77.66%':>7} {'-0.20%':>7} {'145.3x':>8}")
print(f"    {'GPT-2 Transformer':<20} {baseline_acc:>6.2f}% {best['post_train_accuracy']:>6.2f}% "
      f"{best['gap_to_baseline']:>+6.2f}% {best['param_compression']:>7.1f}x")

pct = best['post_train_accuracy'] / baseline_acc * 100
if pct >= 98:
    verdict = "✅ TESE VALIDADA EM MLP + CNN + TRANSFORMER!"
elif pct >= 95:
    verdict = "✅ Codebook funciona em Transformer"
elif pct >= 90:
    verdict = "⚠️  Recovery moderado"
else:
    verdict = "❌ Recovery insuficiente"
print(f"\n  {verdict} ({pct:.1f}% do baseline)")

# Todas as configs
print(f"\n  {'K':>5} {'B':>3} {'Pre%':>7} {'Post%':>7} {'Best%':>7} {'Params':>8} {'Comp':>7} {'Layers':>7}")
print("  " + "-" * 60)
for e in learning_results["experiments"]:
    print(f"  {e['K']:>5} {e['block_size']:>3} {e['pre_train_accuracy']:>7.2f} "
          f"{e['post_train_accuracy']:>7.2f} {e['best_epoch_accuracy']:>7.2f} "
          f"{e['trainable_params']:>8,} {e['param_compression']:>6.1f}x {e['layers_replaced']:>7}")

print(f"\n💾 Resultados salvos em: {out_path}")
print(f"\n📋 Para copiar resultados:")
print(f"   cat {out_path}")
