# ============================================================
# 🧬 Tensor-Vivo — Exp5: Codebook Learning em GPT-2 Small
# Google Colab — A100 GPU
# ============================================================
# Cole cada seção (# %%) como uma célula separada no Colab.
# Tempo estimado total: ~15-25 minutos com A100.
# ============================================================

# %% ════════════════════════════════════════════════════════════
# CÉLULA 1: Setup + Instalar dependências
# ════════════════════════════════════════════════════════════
!pip install -q transformers datasets accelerate peft

import os, json, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import MiniBatchKMeans, KMeans

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

DATA_DIR = "/content/tensor_vivo_exp5"
os.makedirs(DATA_DIR, exist_ok=True)

# %% ════════════════════════════════════════════════════════════
# CÉLULA 2: Fine-tune GPT-2 no SST-2 → Baseline
# ════════════════════════════════════════════════════════════
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer
from datasets import load_dataset

print("=" * 70)
print("  FASE 1: Fine-tune GPT-2 Small no SST-2 (Baseline)")
print("=" * 70)

# ── Tokenizer ──
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# ── Dataset ──
dataset = load_dataset("glue", "sst2")

def tokenize_fn(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length",
                     max_length=128, return_tensors=None)

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["sentence", "idx"])
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch")

train_loader = torch.utils.data.DataLoader(
    tokenized["train"], batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(
    tokenized["validation"], batch_size=64)

# ── Modelo ──
model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
model.config.pad_token_id = tokenizer.pad_token_id
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
transformer_params = sum(p.numel() for n, p in model.named_parameters()
                         if "transformer.h." in n and ("attn" in n or "mlp" in n)
                         and "weight" in n and p.dim() == 2)
print(f"\nTotal params: {total_params:,}")
print(f"Transformer linear params (alvos): {transformer_params:,}")

def evaluate_model(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total * 100

# ── Fine-tune ──
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

print(f"\nFine-tuning GPT-2 no SST-2:")
for epoch in range(1, 4):
    model.train()
    total_loss = 0
    t0 = time.time()
    for i, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

        if (i + 1) % 200 == 0:
            print(f"    Step {i+1}/{len(train_loader)}: loss={total_loss/(i+1):.4f}")

    avg_loss = total_loss / len(train_loader)
    acc = evaluate_model(model, val_loader)
    elapsed = time.time() - t0
    print(f"  Epoch {epoch}: loss={avg_loss:.4f}  val_acc={acc:.2f}%  ({elapsed:.1f}s)")

# ── Salvar baseline ──
torch.save(model.state_dict(), f"{DATA_DIR}/gpt2_sst2_baseline.pt")
baseline_acc = evaluate_model(model, val_loader)
print(f"\n📊 Baseline GPT-2 SST-2: {baseline_acc:.2f}% accuracy")
print(f"💾 Modelo salvo!")

# %% ════════════════════════════════════════════════════════════
# CÉLULA 3: Codebook Quantization Grid (9 combinações)
# ════════════════════════════════════════════════════════════
print("=" * 70)
print("  FASE 2: Codebook Quantization (sem treino)")
print("=" * 70)

def quantize_layer(weight_tensor, K, block_size):
    """Quantiza pesos via K-Means Flatten+Chunk."""
    w = weight_tensor.detach().cpu().numpy().flatten()
    n = len(w)
    remainder = n % block_size
    if remainder:
        w = np.concatenate([w, np.zeros(block_size - remainder)])
    blocks = w.reshape(-1, block_size)
    actual_K = min(K, blocks.shape[0])
    km = MiniBatchKMeans(n_clusters=actual_K, random_state=42,
                         batch_size=min(1000, blocks.shape[0]))
    km.fit(blocks)
    qf = km.cluster_centers_[km.labels_].flatten()[:n]
    mse = float(np.mean((weight_tensor.detach().cpu().numpy().flatten() - qf) ** 2))
    orig_b = n * 4
    comp_b = actual_K * block_size * 4 + blocks.shape[0] * 2
    return qf, mse, orig_b, comp_b, actual_K

# Layers to quantize (attention + MLP linear weights only)
TARGET_SUFFIXES = ["attn.c_attn.weight", "attn.c_proj.weight",
                   "mlp.c_fc.weight", "mlp.c_proj.weight"]

def is_target_layer(name):
    return any(name.endswith(s) for s in TARGET_SUFFIXES)

K_values = [256, 512, 1024]
block_sizes = [16, 32, 64]

quant_results = {"baseline_accuracy": baseline_acc, "total_params": total_params,
                 "target_params": transformer_params, "experiments": []}

print(f"\n{'K':>6} {'Block':>6} {'Acc%':>8} {'Loss':>8} {'Ratio':>8} {'Attn MSE':>12} {'MLP MSE':>12}")
print("-" * 80)

for bs in block_sizes:
    for K in K_values:
        # Reload fresh model
        m = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
        m.config.pad_token_id = tokenizer.pad_token_id
        m.load_state_dict(torch.load(f"{DATA_DIR}/gpt2_sst2_baseline.pt", weights_only=True))
        m = m.to(device)

        tot_orig = tot_comp = 0
        attn_mse = mlp_mse = 0; attn_n = mlp_n = 0

        for name, param in m.named_parameters():
            if not is_target_layer(name): continue
            qf, mse, ob, cb, aK = quantize_layer(param, K, bs)
            qt = torch.tensor(qf[:param.numel()], dtype=torch.float32).reshape(param.shape).to(device)
            with torch.no_grad(): param.copy_(qt)
            tot_orig += ob; tot_comp += cb
            if "attn" in name: attn_mse += mse; attn_n += 1
            else: mlp_mse += mse; mlp_n += 1

        acc = evaluate_model(m, val_loader)
        ratio = tot_orig / tot_comp if tot_comp else 0
        am = attn_mse / attn_n if attn_n else 0
        mm = mlp_mse / mlp_n if mlp_n else 0
        print(f"{K:>6} {bs:>6} {acc:>8.2f} {baseline_acc-acc:>+7.2f}% {ratio:>7.1f}x {am:>12.8f} {mm:>12.8f}")

        quant_results["experiments"].append({
            "K": K, "block_size": bs, "accuracy": round(acc, 4),
            "accuracy_loss": round(baseline_acc - acc, 4),
            "compression_ratio": round(ratio, 2),
            "attn_mse": round(am, 10), "mlp_mse": round(mm, 10),
        })
        del m; torch.cuda.empty_cache()

with open(f"{DATA_DIR}/exp5_quantize_results.json", "w") as f:
    json.dump(quant_results, f, indent=2)
print(f"\n💾 Salvo: exp5_quantize_results.json")

# %% ════════════════════════════════════════════════════════════
# CÉLULA 4: CodebookLinear + CodebookGPT2
# ════════════════════════════════════════════════════════════

class CodebookLinear(nn.Module):
    """Linear layer with weights from a learned codebook."""
    def __init__(self, original_linear, K, block_size):
        super().__init__()
        assert isinstance(original_linear, nn.Linear), \
            f"Expected nn.Linear, got {type(original_linear)}"
        self.out_features = original_linear.out_features
        self.in_features = original_linear.in_features
        self.block_size = block_size

        w = original_linear.weight.detach().cpu().numpy().flatten()
        n = len(w); self.original_numel = n
        if n % block_size:
            w = np.concatenate([w, np.zeros(block_size - n % block_size)])

        blocks = w.reshape(-1, block_size)
        actual_K = min(K, blocks.shape[0])

        km = MiniBatchKMeans(n_clusters=actual_K, random_state=42,
                             batch_size=min(1000, blocks.shape[0]))
        km.fit(blocks)

        self.codebook = nn.Parameter(
            torch.tensor(km.cluster_centers_, dtype=torch.float32))
        self.register_buffer(
            'indices', torch.tensor(km.labels_, dtype=torch.long))

        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.detach().clone())
        else:
            self.bias = None

        self.actual_K = actual_K
        self.num_blocks = blocks.shape[0]

    def forward(self, x):
        W = self.codebook[self.indices].reshape(-1)[:self.original_numel]
        W = W.reshape(self.out_features, self.in_features)
        return F.linear(x, W, self.bias)


def replace_linear_with_codebook(model, K, block_size):
    """Replace target linear layers in GPT-2 with CodebookLinear."""
    replacements = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and is_target_layer(name + ".weight"):
            replacements[name] = module

    for name, original in replacements.items():
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        cb_layer = CodebookLinear(original, K, block_size)
        setattr(parent, parts[-1], cb_layer)

    return model


def count_trainable(model):
    """Count trainable params (codebook + biases + LayerNorm)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_codebook_info(model):
    """Extract per-layer codebook info."""
    info = []
    for name, module in model.named_modules():
        if isinstance(module, CodebookLinear):
            info.append({
                "name": name,
                "K": module.actual_K,
                "blocks": module.num_blocks,
                "block_size": module.block_size,
                "cb_params": module.actual_K * module.block_size,
                "bias_params": module.bias.numel() if module.bias is not None else 0,
            })
    return info

print("✅ CodebookLinear e replace_linear_with_codebook definidos")

# %% ════════════════════════════════════════════════════════════
# CÉLULA 5: Codebook Learning — 4 configs × 5 epochs
# ════════════════════════════════════════════════════════════
print("=" * 70)
print("  FASE 3: CODEBOOK LEARNING — Treinar APENAS o codebook do GPT-2")
print("=" * 70)

configs = [
    (256, 16),   # Granular — mais centróides
    (512, 16),   # Match com quantization
    (512, 32),   # Compressão intermediária
    (1024, 32),  # Alta capacidade
]

learning_results = {
    "baseline_accuracy": baseline_acc,
    "total_params": total_params,
    "target_params": transformer_params,
    "model": "GPT2ForSequenceClassification",
    "task": "SST-2 (sentiment)",
    "experiments": [],
}

for K, block_size in configs:
    print(f"\n{'━'*70}")
    print(f"  K={K}, Block={block_size}")
    print(f"{'━'*70}")

    # ── Rebuild model and replace ──
    model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.load_state_dict(torch.load(f"{DATA_DIR}/gpt2_sst2_baseline.pt", weights_only=True))
    model = replace_linear_with_codebook(model, K, block_size)
    model = model.to(device)

    # ── Freeze everything except codebook + biases + LayerNorm + classifier ──
    for name, param in model.named_parameters():
        param.requires_grad = False  # freeze all first

    for name, param in model.named_parameters():
        if "codebook" in name:
            param.requires_grad = True
        elif "bias" in name:
            param.requires_grad = True
        elif "ln_" in name or "ln_f" in name:
            param.requires_grad = True
        elif "score" in name:  # classification head
            param.requires_grad = True

    pre_acc = evaluate_model(model, val_loader)
    trainable = count_trainable(model)
    compression = total_params / trainable if trainable else 0
    cb_info = get_codebook_info(model)

    # Summarize
    total_cb_params = sum(l["cb_params"] for l in cb_info)
    total_replaced = len(cb_info)
    print(f"  Pré-treino: {pre_acc:.2f}%")
    print(f"  Layers replaced: {total_replaced}")
    print(f"  Codebook params: {total_cb_params:,}")
    print(f"  Total trainable: {trainable:,} ({compression:.1f}x menos)")

    # Top layers info
    for li in cb_info[:4]:
        print(f"    {li['name']:<45} K={li['K']:>5} blocks={li['blocks']:>7} cb={li['cb_params']:>8,}")
    if len(cb_info) > 4:
        print(f"    ... and {len(cb_info)-4} more layers")

    print(f"\n  {'Epoch':>6} {'Acc%':>8} {'Loss':>10} {'Δ Pre':>9} {'Δ Base':>9}")
    print("  " + "-" * 45)

    # ── Training ──
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    criterion = nn.CrossEntropyLoss()
    epochs_data = []

    t0 = time.time()
    for epoch in range(1, 6):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        acc = evaluate_model(model, val_loader)
        dp = acc - pre_acc; db = acc - baseline_acc
        print(f"  {epoch:>6} {acc:>8.2f} {avg_loss:>10.4f} {dp:>+8.2f}% {db:>+8.2f}%")
        epochs_data.append({
            "epoch": epoch, "accuracy": round(acc, 4),
            "loss": round(avg_loss, 6),
            "delta_pre": round(dp, 4), "delta_base": round(db, 4)
        })

    elapsed = time.time() - t0
    final_acc = epochs_data[-1]["accuracy"]
    best_acc = max(e["accuracy"] for e in epochs_data)
    print(f"\n  📊 Resumo: pré={pre_acc:.2f}% → pós={final_acc:.2f}% (best={best_acc:.2f}%)")
    print(f"     Params: {trainable:,} ({compression:.1f}x) | Tempo: {elapsed:.1f}s")

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
        "layers_replaced": total_replaced,
        "training_time_seconds": round(elapsed, 1),
        "epochs": epochs_data,
    })
    del model; torch.cuda.empty_cache()

with open(f"{DATA_DIR}/exp5_learning_results.json", "w") as f:
    json.dump(learning_results, f, indent=2)
print(f"\n💾 Salvo: exp5_learning_results.json")

# %% ════════════════════════════════════════════════════════════
# CÉLULA 6: Comparação com LoRA (mesmo nº de params)
# ════════════════════════════════════════════════════════════
print("=" * 70)
print("  FASE 4: Comparação com LoRA (mesmos params treináveis)")
print("=" * 70)

from peft import get_peft_model, LoraConfig, TaskType

# Pick best codebook config
best_cb = max(learning_results["experiments"], key=lambda x: x["post_train_accuracy"])
target_params = best_cb["trainable_params"]

# Try different LoRA ranks to match param count
lora_results = {"target_params_from_codebook": target_params, "experiments": []}

for rank in [2, 4, 8, 16]:
    print(f"\n  LoRA rank={rank}:")

    base_model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.load_state_dict(
        torch.load(f"{DATA_DIR}/gpt2_sst2_baseline.pt", weights_only=True))

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],
        bias="none",
    )
    lora_model = get_peft_model(base_model, lora_config)
    lora_model = lora_model.to(device)

    lora_trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f"    Trainable params: {lora_trainable:,}")

    # Train LoRA
    optimizer = optim.AdamW(
        [p for p in lora_model.parameters() if p.requires_grad],
        lr=5e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    criterion = nn.CrossEntropyLoss()

    t0 = time.time()
    for epoch in range(1, 6):
        lora_model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = lora_model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

    lora_acc = evaluate_model(lora_model, val_loader)
    elapsed = time.time() - t0
    print(f"    Accuracy: {lora_acc:.2f}% | Time: {elapsed:.1f}s")

    lora_results["experiments"].append({
        "rank": rank,
        "trainable_params": lora_trainable,
        "accuracy": round(lora_acc, 4),
        "training_time_seconds": round(elapsed, 1),
    })
    del lora_model, base_model; torch.cuda.empty_cache()

with open(f"{DATA_DIR}/exp5_lora_comparison.json", "w") as f:
    json.dump(lora_results, f, indent=2)
print(f"\n💾 Salvo: exp5_lora_comparison.json")

# %% ════════════════════════════════════════════════════════════
# CÉLULA 7: Veredicto Final + Download
# ════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  🏆 VEREDICTO — CODEBOOK LEARNING EM GPT-2 TRANSFORMER")
print(f"{'='*70}")

best = max(learning_results["experiments"], key=lambda x: x["post_train_accuracy"])
print(f"\n  Baseline GPT-2 SST-2:  {baseline_acc:.2f}%  ({total_params:,} params)")
print(f"\n  Melhor Codebook:")
print(f"    K={best['K']}, Block={best['block_size']}")
print(f"    Accuracy:  {best['post_train_accuracy']:.2f}% (best: {best['best_epoch_accuracy']:.2f}%)")
print(f"    Params:    {best['trainable_params']:,} ({best['param_compression']:.1f}x menos)")
print(f"    Gap:       {best['gap_to_baseline']:.2f}%")

# vs LoRA
print(f"\n  📊 Codebook vs LoRA:")
print(f"    {'Método':<25} {'Params':>10} {'Accuracy':>10}")
print("    " + "-" * 50)
print(f"    {'Baseline (full model)':<25} {total_params:>10,} {baseline_acc:>9.2f}%")
for e in learning_results["experiments"]:
    label = f"Codebook K={e['K']} B={e['block_size']}"
    print(f"    {label:<25} {e['trainable_params']:>10,} {e['post_train_accuracy']:>9.2f}%")
for e in lora_results["experiments"]:
    label = f"LoRA rank={e['rank']}"
    print(f"    {label:<25} {e['trainable_params']:>10,} {e['accuracy']:>9.2f}%")

# ── Comparação cross-experiment ──
print(f"\n  📊 Evolução da Tese (todos os experimentos):")
print(f"    {'Exp':<20} {'Baseline':>8} {'Codebook':>9} {'Gap':>6} {'Compress':>10}")
print("    " + "-" * 55)
print(f"    {'MNIST MLP':<20} {'97.53%':>8} {'97.56%':>9} {'0.03%':>6} {'40.8x':>10}")
print(f"    {'CIFAR-10 CNN':<20} {'77.86%':>8} {'77.66%':>9} {'0.20%':>6} {'145.3x':>10}")
print(f"    {'GPT-2 Transformer':<20} {baseline_acc:>7.2f}% {best['post_train_accuracy']:>8.2f}% "
      f"{best['gap_to_baseline']:>5.2f}% {best['param_compression']:>9.1f}x")

pct = best['post_train_accuracy'] / baseline_acc * 100
if pct >= 98:
    print(f"\n  ✅ CODEBOOK LEARNING FUNCIONA EM TRANSFORMERS! ({pct:.1f}% do baseline)")
    print(f"     A TESE ESTÁ VALIDADA EM MLP + CNN + TRANSFORMER.")
elif pct >= 95:
    print(f"\n  ✅ Codebook funciona parcialmente em Transformer ({pct:.1f}%)")
elif pct >= 90:
    print(f"\n  ⚠️ Recovery moderado em Transformer ({pct:.1f}%)")
else:
    print(f"\n  ❌ Recovery insuficiente em Transformer ({pct:.1f}%)")

# ── Download ──
from google.colab import files
for fname in ["exp5_quantize_results.json", "exp5_learning_results.json", "exp5_lora_comparison.json"]:
    fpath = f"{DATA_DIR}/{fname}"
    if os.path.exists(fpath):
        files.download(fpath)
print("\n✅ Arquivos baixados! Cole-os em pesquisas/tensor-vivo/dados/")
