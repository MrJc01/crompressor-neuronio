# 🧬 Tensor-Vivo — Exp5 CORRIGIDO: Codebook Learning em GPT-2 Small
# Google Colab — A100 GPU
# ============================================================
# CORREÇÃO: GPT-2 usa Conv1D, não nn.Linear!
# Este script detecta e substitui Conv1D corretamente.
# ============================================================

# %% ════════════════════════════════════════════════════════════
# CÉLULA 1: Setup
# ════════════════════════════════════════════════════════════
!pip install -q transformers datasets accelerate peft

import os, json, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import MiniBatchKMeans
from transformers.pytorch_utils import Conv1D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

DATA_DIR = "/content/tensor_vivo_exp5"
os.makedirs(DATA_DIR, exist_ok=True)

# %% ════════════════════════════════════════════════════════════
# CÉLULA 2: Carregar baseline do Drive (ou treinar se não existir)
# ════════════════════════════════════════════════════════════
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer
from datasets import load_dataset

print("=" * 70)
print("  FASE 1: Carregar/Treinar GPT-2 Baseline para SST-2")
print("=" * 70)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

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

# Tentar carregar baseline do Drive
DRIVE_DIR = "/content/drive/MyDrive/tensor-vivo-resultados"
baseline_path = f"{DATA_DIR}/gpt2_sst2_baseline.pt"
drive_path = f"{DRIVE_DIR}/gpt2_sst2_baseline.pt"

model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
model.config.pad_token_id = tokenizer.pad_token_id

if os.path.exists(drive_path):
    print("  📂 Carregando baseline do Drive...")
    model.load_state_dict(torch.load(drive_path, weights_only=True, map_location=device))
    model = model.to(device)
    baseline_acc = evaluate_model(model, val_loader)
    print(f"  ✅ Baseline carregado: {baseline_acc:.2f}%")
    torch.save(model.state_dict(), baseline_path)
elif os.path.exists(baseline_path):
    print("  📂 Carregando baseline local...")
    model.load_state_dict(torch.load(baseline_path, weights_only=True, map_location=device))
    model = model.to(device)
    baseline_acc = evaluate_model(model, val_loader)
    print(f"  ✅ Baseline carregado: {baseline_acc:.2f}%")
else:
    print("  🔧 Treinando baseline do zero (3 epochs)...")
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, 4):
        model.train()
        total_loss = 0; t0 = time.time()
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
        acc = evaluate_model(model, val_loader)
        print(f"    Epoch {epoch}: loss={total_loss/len(train_loader):.4f} acc={acc:.2f}% ({time.time()-t0:.0f}s)")
    torch.save(model.state_dict(), baseline_path)
    baseline_acc = evaluate_model(model, val_loader)
    print(f"  ✅ Baseline treinado: {baseline_acc:.2f}%")

total_params = sum(p.numel() for p in model.parameters())

# Contar CORRETAMENTE os params alvo (Conv1D, não nn.Linear)
TARGET_SUFFIXES = ["attn.c_attn.weight", "attn.c_proj.weight",
                   "mlp.c_fc.weight", "mlp.c_proj.weight"]

def is_target_layer_name(name):
    return any(name.endswith(s) for s in TARGET_SUFFIXES)

target_params = sum(p.numel() for n, p in model.named_parameters() if is_target_layer_name(n))
print(f"\n  Total params:  {total_params:,}")
print(f"  Target params: {target_params:,} ({target_params/total_params*100:.1f}%)")

# Verificar tipos de camadas
print("\n  Tipos de camadas no GPT-2:")
for name, module in model.named_modules():
    if hasattr(module, 'weight') and any(s.replace('.weight','') in name for s in TARGET_SUFFIXES):
        print(f"    {name}: {type(module).__name__} → weight shape {module.weight.shape}")
        break

# %% ════════════════════════════════════════════════════════════
# CÉLULA 3: CodebookConv1D (CORRIGIDO para GPT-2!)
# ════════════════════════════════════════════════════════════

class CodebookConv1D(nn.Module):
    """Substitui Conv1D do GPT-2 por versão com codebook treinável.

    NOTA: Conv1D do HuggingFace é similar a nn.Linear mas com peso transposto:
    Conv1D: weight shape = (in_features, out_features), forward = x @ weight + bias
    nn.Linear: weight shape = (out_features, in_features), forward = x @ weight.T + bias
    """
    def __init__(self, original_conv1d, K, block_size):
        super().__init__()
        assert isinstance(original_conv1d, Conv1D), \
            f"Expected Conv1D, got {type(original_conv1d)}"

        self.nf = original_conv1d.nf  # out_features
        self.weight_shape = original_conv1d.weight.shape  # (in_f, out_f)
        self.block_size = block_size

        w = original_conv1d.weight.detach().cpu().numpy().flatten()
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

        if original_conv1d.bias is not None:
            self.bias = nn.Parameter(original_conv1d.bias.detach().clone())
        else:
            self.bias = None

        self.actual_K = actual_K
        self.num_blocks = blocks.shape[0]

    def forward(self, x):
        W = self.codebook[self.indices].reshape(-1)[:self.original_numel]
        W = W.reshape(self.weight_shape)
        # Conv1D forward: x @ weight + bias (NOT transposed like nn.Linear)
        out = torch.matmul(x, W)
        if self.bias is not None:
            out = out + self.bias
        return out


def replace_conv1d_with_codebook(model, K, block_size):
    """Replace ALL target Conv1D layers in GPT-2 with CodebookConv1D."""
    replacements = {}
    for name, module in model.named_modules():
        if isinstance(module, Conv1D):
            # Só substituir attention e MLP, não classification head
            if any(target_name in name for target_name in
                   ["c_attn", "c_proj", "c_fc"]):
                replacements[name] = module

    print(f"  🔧 Encontradas {len(replacements)} camadas Conv1D para substituir")

    for name, original in replacements.items():
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            if p.isdigit():
                parent = parent[int(p)]
            else:
                parent = getattr(parent, p)
        cb_layer = CodebookConv1D(original, K, block_size)
        setattr(parent, parts[-1], cb_layer)
        # Verificar que foi substituído
        check = getattr(parent, parts[-1])
        assert isinstance(check, CodebookConv1D), f"Falha ao substituir {name}"

    return model, len(replacements)


def get_codebook_info(model):
    info = []
    for name, module in model.named_modules():
        if isinstance(module, CodebookConv1D):
            info.append({
                "name": name, "K": module.actual_K,
                "blocks": module.num_blocks, "block_size": module.block_size,
                "cb_params": module.actual_K * module.block_size,
                "bias_params": module.bias.numel() if module.bias is not None else 0,
                "weight_shape": list(module.weight_shape),
            })
    return info

print("✅ CodebookConv1D definido (corrigido para GPT-2 Conv1D)")

# %% ════════════════════════════════════════════════════════════
# CÉLULA 4: Codebook Learning — 4 configs × 5 epochs
# ════════════════════════════════════════════════════════════
print("=" * 70)
print("  FASE 2: CODEBOOK LEARNING — Treinar APENAS codebooks do GPT-2")
print("  (CORRIGIDO: agora substitui Conv1D corretamente)")
print("=" * 70)

configs = [
    (256, 16),   # Granular
    (512, 16),   # Match quantization
    (512, 32),   # Compressão intermediária
    (1024, 32),  # Alta capacidade
]

learning_results = {
    "baseline_accuracy": baseline_acc,
    "total_params": total_params,
    "target_params": target_params,
    "model": "GPT2ForSequenceClassification",
    "task": "SST-2 (sentiment)",
    "bug_fix": "Conv1D detection — v2",
    "experiments": [],
}

for K, block_size in configs:
    print(f"\n{'━'*70}")
    print(f"  K={K}, Block={block_size}")
    print(f"{'━'*70}")

    # Rebuild fresh model
    mdl = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
    mdl.config.pad_token_id = tokenizer.pad_token_id
    mdl.load_state_dict(torch.load(baseline_path, weights_only=True, map_location="cpu"))

    # Replace Conv1D with CodebookConv1D
    mdl, n_replaced = replace_conv1d_with_codebook(mdl, K, block_size)
    mdl = mdl.to(device)

    # Freeze all, then unfreeze codebook + biases + LayerNorm + classifier
    for param in mdl.parameters():
        param.requires_grad = False
    for name, param in mdl.named_parameters():
        if "codebook" in name:
            param.requires_grad = True
        elif "bias" in name:
            param.requires_grad = True
        elif "ln_" in name or "ln_f" in name:
            param.requires_grad = True
        elif "score" in name:
            param.requires_grad = True

    pre_acc = evaluate_model(mdl, val_loader)
    trainable = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    compression = total_params / trainable if trainable else 0
    cb_info = get_codebook_info(mdl)
    total_cb_params = sum(l["cb_params"] for l in cb_info)

    print(f"  Layers replaced: {n_replaced} (deve ser 48 = 4 × 12 blocos)")
    print(f"  Codebook params: {total_cb_params:,}")
    print(f"  Total trainable: {trainable:,} ({compression:.1f}x menos)")
    print(f"  Pré-treino accuracy: {pre_acc:.2f}%")

    # Print first 4 layers
    for li in cb_info[:4]:
        print(f"    {li['name']:<45} K={li['K']:>5} blocks={li['blocks']:>7} "
              f"cb={li['cb_params']:>8,} shape={li['weight_shape']}")
    if len(cb_info) > 4:
        print(f"    ... e mais {len(cb_info)-4} camadas")

    print(f"\n  {'Epoch':>6} {'Acc%':>8} {'Loss':>10} {'Δ Pre':>9} {'Δ Base':>9}")
    print("  " + "-" * 45)

    optimizer = optim.AdamW(
        [p for p in mdl.parameters() if p.requires_grad], lr=5e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    criterion = nn.CrossEntropyLoss()
    epochs_data = []

    t0 = time.time()
    for epoch in range(1, 6):
        mdl.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = mdl(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        acc = evaluate_model(mdl, val_loader)
        dp = acc - pre_acc; db = acc - baseline_acc
        print(f"  {epoch:>6} {acc:>8.2f} {avg_loss:>10.4f} {dp:>+8.2f}% {db:>+8.2f}%")
        epochs_data.append({"epoch": epoch, "accuracy": round(acc, 4),
                            "loss": round(avg_loss, 6),
                            "delta_pre": round(dp, 4), "delta_base": round(db, 4)})
    elapsed = time.time() - t0

    final_acc = epochs_data[-1]["accuracy"]
    best_acc_config = max(e["accuracy"] for e in epochs_data)
    print(f"\n  📊 pré={pre_acc:.2f}% → pós={final_acc:.2f}% (best={best_acc_config:.2f}%)")
    print(f"     Params: {trainable:,} ({compression:.1f}x) | Tempo: {elapsed:.1f}s")

    learning_results["experiments"].append({
        "K": K, "block_size": block_size,
        "pre_train_accuracy": round(pre_acc, 4),
        "post_train_accuracy": final_acc,
        "best_epoch_accuracy": best_acc_config,
        "recovery": round(final_acc - pre_acc, 4),
        "gap_to_baseline": round(baseline_acc - final_acc, 4),
        "trainable_params": trainable,
        "param_compression": round(compression, 2),
        "codebook_params": total_cb_params,
        "layers_replaced": n_replaced,
        "training_time_seconds": round(elapsed, 1),
        "epochs": epochs_data,
    })
    del mdl; torch.cuda.empty_cache()

with open(f"{DATA_DIR}/exp5_learning_results_v2.json", "w") as f:
    json.dump(learning_results, f, indent=2)
print(f"\n💾 Salvo: exp5_learning_results_v2.json")

# %% ════════════════════════════════════════════════════════════
# CÉLULA 5: Veredicto + Comparação com LoRA (resultados v1)
# ════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  🏆 VEREDICTO v2 — CODEBOOK LEARNING EM GPT-2 (Conv1D corrigido)")
print(f"{'='*70}")

best = max(learning_results["experiments"], key=lambda x: x["post_train_accuracy"])

print(f"\n  Baseline GPT-2 SST-2: {baseline_acc:.2f}% ({total_params:,} params)")
print(f"\n  Melhor Codebook:")
print(f"    K={best['K']}, Block={best['block_size']}")
print(f"    Accuracy:  {best['post_train_accuracy']:.2f}% (best epoch: {best['best_epoch_accuracy']:.2f}%)")
print(f"    Params:    {best['trainable_params']:,} ({best['param_compression']:.1f}x menos)")
print(f"    Codebook params: {best['codebook_params']:,}")
print(f"    Layers replaced: {best['layers_replaced']}")

# Carregar LoRA resultados v1
lora_path = f"{DATA_DIR}/exp5_lora_comparison.json"
if os.path.exists(lora_path):
    with open(lora_path) as f:
        lora_results = json.load(f)
    print(f"\n  📊 Codebook vs LoRA:")
    print(f"    {'Método':<30} {'Params':>10} {'Accuracy':>10}")
    print("    " + "-" * 55)
    print(f"    {'Baseline (full)':<30} {total_params:>10,} {baseline_acc:>9.2f}%")
    for e in learning_results["experiments"]:
        label = f"CB K={e['K']} B={e['block_size']} (Conv1D)"
        print(f"    {label:<30} {e['trainable_params']:>10,} {e['post_train_accuracy']:>9.2f}%")
    for e in lora_results["experiments"]:
        label = f"LoRA rank={e['rank']}"
        print(f"    {label:<30} {e['trainable_params']:>10,} {e['accuracy']:>9.2f}%")

# Cross-experiment
print(f"\n  📊 Evolução da Tese:")
print(f"    {'Exp':<20} {'Baseline':>8} {'Codebook':>9} {'Gap':>7} {'Compress':>10}")
print("    " + "-" * 58)
print(f"    {'MNIST MLP':<20} {'97.53%':>8} {'97.56%':>9} {'+0.03%':>7} {'40.8x':>10}")
print(f"    {'CIFAR-10 CNN':<20} {'77.86%':>8} {'77.66%':>9} {'-0.20%':>7} {'145.3x':>10}")
print(f"    {'GPT-2 Transformer':<20} {baseline_acc:>7.2f}% {best['post_train_accuracy']:>8.2f}% "
      f"{best['gap_to_baseline']:>+6.2f}% {best['param_compression']:>9.1f}x")

pct = best['post_train_accuracy'] / baseline_acc * 100
if pct >= 98:
    print(f"\n  ✅ CODEBOOK LEARNING FUNCIONA EM TRANSFORMERS! ({pct:.1f}%)")
    print(f"     TESE VALIDADA: MLP + CNN + TRANSFORMER.")
elif pct >= 95:
    print(f"\n  ✅ Codebook funciona em Transformer ({pct:.1f}% do baseline)")
elif pct >= 90:
    print(f"\n  ⚠️ Recovery moderado ({pct:.1f}% do baseline)")
else:
    print(f"\n  ❌ Recovery insuficiente ({pct:.1f}% do baseline)")

# %% ════════════════════════════════════════════════════════════
# CÉLULA 6: Salvar no Drive
# ════════════════════════════════════════════════════════════
import shutil
from google.colab import drive

try:
    drive.mount('/content/drive')
except:
    pass

DRIVE_DIR = "/content/drive/MyDrive/tensor-vivo-resultados"
os.makedirs(DRIVE_DIR, exist_ok=True)

for fname in ["exp5_learning_results_v2.json", "exp5_quantize_results.json",
              "exp5_lora_comparison.json", "gpt2_sst2_baseline.pt"]:
    src = f"{DATA_DIR}/{fname}"
    dst = f"{DRIVE_DIR}/{fname}"
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  ✅ {fname} → Drive ({os.path.getsize(dst)/1024:.1f} KB)")

from google.colab import files
for fname in ["exp5_learning_results_v2.json"]:
    fpath = f"{DATA_DIR}/{fname}"
    if os.path.exists(fpath):
        files.download(fpath)

print(f"\n📁 Salvo no Drive: {DRIVE_DIR}")
print("✅ Pronto! Copie exp5_learning_results_v2.json para pesquisas/tensor-vivo/dados/")
