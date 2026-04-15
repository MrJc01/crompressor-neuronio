# ============================================================
# 🧬 Tensor-Vivo — Exp3: CIFAR-10 CNN Codebook Learning
# Google Colab — A100 GPU
# ============================================================
# Cole cada seção como uma célula separada no Colab.
# Tempo estimado total: ~5-10 minutos com A100.
# ============================================================

# %% [markdown]
# # 🧬 Tensor-Vivo — Exp3: Codebook Learning em CNN CIFAR-10
# > **Pergunta:** O Codebook Learning que funcionou em MNIST MLP funciona em CNN?

# %% ════════════════════════════════════════════════════════════
# CÉLULA 1: Setup
# ════════════════════════════════════════════════════════════
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import MiniBatchKMeans, KMeans
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

DATA_DIR = "/content/tensor_vivo_data"
os.makedirs(DATA_DIR, exist_ok=True)

# %% ════════════════════════════════════════════════════════════
# CÉLULA 2: Modelo CNN + Treino Baseline
# ════════════════════════════════════════════════════════════
class SimpleCNN(nn.Module):
    """CNN para CIFAR-10: Conv(3→32)→Conv(32→64)→FC(4096→256→10)"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total * 100

# ── Dados ──
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

train_ds = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_transform)
test_ds = datasets.CIFAR10(DATA_DIR, train=False, transform=test_transform)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1000, num_workers=2)

# ── Treinar Baseline ──
print("=" * 60)
print("  FASE 1: Treinar CNN Baseline")
print("=" * 60)

model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
criterion = nn.CrossEntropyLoss()

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal params: {total_params:,}")

best_acc = 0
for epoch in range(1, 31):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    scheduler.step()
    acc = evaluate(model, test_loader)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), f"{DATA_DIR}/cifar_cnn.pt")
    print(f"  Epoch {epoch:2d}: accuracy={acc:.2f}%")

model.load_state_dict(torch.load(f"{DATA_DIR}/cifar_cnn.pt", weights_only=True))
baseline_acc = evaluate(model, test_loader)
baseline_params = total_params
print(f"\n📊 Baseline: {baseline_acc:.2f}% accuracy, {baseline_params:,} params")

# %% ════════════════════════════════════════════════════════════
# CÉLULA 3: Codebook Quantization Grid (15 combinações)
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("  FASE 2: Codebook Quantization (sem treino)")
print("=" * 60)

def quantize_layer(weight_tensor, K, block_size):
    w = weight_tensor.detach().cpu().numpy().flatten()
    n = len(w)
    remainder = n % block_size
    if remainder:
        w = np.concatenate([w, np.zeros(block_size - remainder)])
    blocks = w.reshape(-1, block_size)
    actual_K = min(K, blocks.shape[0])
    km = MiniBatchKMeans(n_clusters=actual_K, random_state=42,
                         batch_size=min(500, blocks.shape[0])) if blocks.shape[0] > 500 \
         else KMeans(n_clusters=actual_K, random_state=42, n_init=3)
    km.fit(blocks)
    qf = km.cluster_centers_[km.labels_].flatten()[:n]
    mse = float(np.mean((weight_tensor.detach().cpu().numpy().flatten() - qf) ** 2))
    orig_b = n * 4
    comp_b = actual_K * block_size * 4 + blocks.shape[0] * 2
    return qf, mse, orig_b, comp_b, actual_K, len(np.unique(km.labels_)) / actual_K * 100

K_values = [32, 64, 128, 256, 512]
block_sizes = [8, 16, 32]
quant_results = {"baseline_accuracy": baseline_acc, "baseline_params": baseline_params, "experiments": []}

print(f"\n{'K':>6} {'Block':>6} {'Acc%':>8} {'Loss':>8} {'Ratio':>8} {'Conv MSE':>12} {'FC MSE':>12}")
print("-" * 80)

for bs in block_sizes:
    for K in K_values:
        m = SimpleCNN().to(device)
        m.load_state_dict(torch.load(f"{DATA_DIR}/cifar_cnn.pt", weights_only=True))
        tot_orig = tot_comp = 0
        conv_mse = fc_mse = 0; conv_n = fc_n = 0
        for name, param in m.named_parameters():
            if "weight" not in name: continue
            qf, mse, ob, cb, aK, util = quantize_layer(param, K, bs)
            qt = torch.tensor(qf[:param.numel()], dtype=torch.float32).reshape(param.shape)
            with torch.no_grad(): param.copy_(qt)
            tot_orig += ob; tot_comp += cb
            if param.dim() == 4: conv_mse += mse; conv_n += 1
            else: fc_mse += mse; fc_n += 1
        acc = evaluate(m, test_loader)
        ratio = tot_orig / tot_comp if tot_comp else 0
        cm = conv_mse / conv_n if conv_n else 0
        fm = fc_mse / fc_n if fc_n else 0
        print(f"{K:>6} {bs:>6} {acc:>8.2f} {baseline_acc-acc:>+7.2f}% {ratio:>7.1f}x {cm:>12.6f} {fm:>12.6f}")
        quant_results["experiments"].append({
            "K": K, "block_size": bs, "accuracy": round(acc, 4),
            "accuracy_loss": round(baseline_acc - acc, 4),
            "compression_ratio": round(ratio, 2),
            "conv_mse": round(cm, 8), "fc_mse": round(fm, 8),
        })

with open(f"{DATA_DIR}/exp3_quantize_results.json", "w") as f:
    json.dump(quant_results, f, indent=2)
print(f"\n💾 Salvo: exp3_quantize_results.json")

# %% ════════════════════════════════════════════════════════════
# CÉLULA 4: CodebookConv2d + CodebookLinear + CodebookCNN
# ════════════════════════════════════════════════════════════

class CodebookConv2d(nn.Module):
    def __init__(self, orig, K, block_size):
        super().__init__()
        self.weight_shape = orig.weight.shape
        self.stride, self.padding = orig.stride, orig.padding
        self.block_size = block_size
        w = orig.weight.detach().cpu().numpy().flatten()
        n = len(w); self.original_numel = n
        if n % block_size: w = np.concatenate([w, np.zeros(block_size - n % block_size)])
        blocks = w.reshape(-1, block_size)
        actual_K = min(K, blocks.shape[0])
        km = MiniBatchKMeans(n_clusters=actual_K, random_state=42,
                             batch_size=min(500, blocks.shape[0])) if blocks.shape[0] > 500 \
             else KMeans(n_clusters=actual_K, random_state=42, n_init=3)
        km.fit(blocks)
        self.codebook = nn.Parameter(torch.tensor(km.cluster_centers_, dtype=torch.float32))
        self.register_buffer('indices', torch.tensor(km.labels_, dtype=torch.long))
        self.bias = nn.Parameter(orig.bias.detach().clone()) if orig.bias is not None else None
        self.actual_K = actual_K; self.num_blocks = blocks.shape[0]

    def forward(self, x):
        W = self.codebook[self.indices].reshape(-1)[:self.original_numel].reshape(self.weight_shape)
        return F.conv2d(x, W, self.bias, self.stride, self.padding)


class CodebookLinear(nn.Module):
    def __init__(self, orig, K, block_size):
        super().__init__()
        self.out_features, self.in_features = orig.out_features, orig.in_features
        self.block_size = block_size
        w = orig.weight.detach().cpu().numpy().flatten()
        n = len(w); self.original_numel = n
        if n % block_size: w = np.concatenate([w, np.zeros(block_size - n % block_size)])
        blocks = w.reshape(-1, block_size)
        actual_K = min(K, blocks.shape[0])
        km = MiniBatchKMeans(n_clusters=actual_K, random_state=42,
                             batch_size=min(500, blocks.shape[0])) if blocks.shape[0] > 500 \
             else KMeans(n_clusters=actual_K, random_state=42, n_init=3)
        km.fit(blocks)
        self.codebook = nn.Parameter(torch.tensor(km.cluster_centers_, dtype=torch.float32))
        self.register_buffer('indices', torch.tensor(km.labels_, dtype=torch.long))
        self.bias = nn.Parameter(orig.bias.detach().clone()) if orig.bias is not None else None
        self.actual_K = actual_K; self.num_blocks = blocks.shape[0]

    def forward(self, x):
        W = self.codebook[self.indices].reshape(-1)[:self.original_numel]
        return F.linear(x, W.reshape(self.out_features, self.in_features), self.bias)


class CodebookCNN(nn.Module):
    def __init__(self, orig, K, block_size):
        super().__init__()
        self.features = nn.Sequential(
            CodebookConv2d(orig.features[0], K, block_size),
            nn.ReLU(), nn.MaxPool2d(2),
            CodebookConv2d(orig.features[3], K, block_size),
            nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            CodebookLinear(orig.classifier[0], K, block_size),
            nn.ReLU(),
            CodebookLinear(orig.classifier[2], K, block_size),
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))

print("✅ CodebookConv2d, CodebookLinear, CodebookCNN definidos")

# %% ════════════════════════════════════════════════════════════
# CÉLULA 5: Codebook Learning — 5 configs × 20 epochs
# ════════════════════════════════════════════════════════════
print("=" * 70)
print("  FASE 3: CODEBOOK LEARNING — Treinar APENAS o codebook")
print("=" * 70)

configs = [
    (128, 8),    # Alta granularidade
    (256, 8),    # Alta granularidade, K alto
    (128, 16),   # Comparação com MNIST Exp2
    (256, 16),   # Match melhor MNIST
    (256, 32),   # Máxima compressão
]

learning_results = {
    "baseline_accuracy": baseline_acc,
    "baseline_params": baseline_params,
    "model": "SimpleCNN_CIFAR10",
    "experiments": [],
}

for K, block_size in configs:
    print(f"\n{'━'*70}")
    print(f"  K={K}, Block={block_size}")
    print(f"{'━'*70}")

    orig = SimpleCNN().to(device)
    orig.load_state_dict(torch.load(f"{DATA_DIR}/cifar_cnn.pt", weights_only=True))
    model = CodebookCNN(orig, K, block_size).to(device)

    pre_acc = evaluate(model, test_loader)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    compression = baseline_params / trainable if trainable else 0

    # Layer info
    layer_info = []
    for name, mod in model.named_modules():
        if isinstance(mod, (CodebookConv2d, CodebookLinear)):
            lt = "Conv2d" if isinstance(mod, CodebookConv2d) else "Linear"
            cb_p = mod.actual_K * mod.block_size
            bias_p = mod.bias.numel() if mod.bias is not None else 0
            layer_info.append({"name": name, "type": lt, "K": mod.actual_K,
                               "blocks": mod.num_blocks, "cb_params": cb_p, "bias_params": bias_p})
            print(f"    {name:<25} {lt:<8} K={mod.actual_K:>4} blocks={mod.num_blocks:>7} cb_params={cb_p:>6}")

    print(f"\n  Pré-treino: {pre_acc:.2f}% | Params: {trainable:,} ({compression:.1f}x menos)")
    print(f"  {'Epoch':>6} {'Acc%':>8} {'Loss':>10} {'Δ Pre':>9} {'Δ Base':>9}")
    print("  " + "-" * 45)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.CrossEntropyLoss()
    epochs_data = []

    t0 = time.time()
    for epoch in range(1, 21):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        acc = evaluate(model, test_loader)
        dp = acc - pre_acc; db = acc - baseline_acc
        print(f"  {epoch:>6} {acc:>8.2f} {avg_loss:>10.4f} {dp:>+8.2f}% {db:>+8.2f}%")
        epochs_data.append({"epoch": epoch, "accuracy": round(acc, 4),
                            "loss": round(avg_loss, 6),
                            "delta_pre": round(dp, 4), "delta_base": round(db, 4)})
    elapsed = time.time() - t0

    final_acc = epochs_data[-1]["accuracy"]
    best_acc_config = max(e["accuracy"] for e in epochs_data)
    print(f"\n  📊 Resumo: pré={pre_acc:.2f}% → pós={final_acc:.2f}% (best={best_acc_config:.2f}%)")
    print(f"     Recovery: {final_acc - pre_acc:+.2f}% | Gap: {baseline_acc - final_acc:.2f}%")
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
        "training_time_seconds": round(elapsed, 1),
        "layer_info": layer_info,
        "epochs": epochs_data,
    })

with open(f"{DATA_DIR}/exp3_learning_results.json", "w") as f:
    json.dump(learning_results, f, indent=2)
print(f"\n💾 Salvo: exp3_learning_results.json")

# %% ════════════════════════════════════════════════════════════
# CÉLULA 6: Veredicto Final
# ════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  🏆 VEREDICTO — CODEBOOK LEARNING EM CNN CIFAR-10")
print(f"{'='*70}")

best = max(learning_results["experiments"], key=lambda x: x["post_train_accuracy"])

print(f"\n  Baseline:      {baseline_acc:.2f}% accuracy, {baseline_params:,} params")
print(f"\n  Melhor resultado:")
print(f"    K={best['K']}, Block={best['block_size']}")
print(f"    Accuracy:    {best['post_train_accuracy']:.2f}% (best epoch: {best['best_epoch_accuracy']:.2f}%)")
print(f"    Params:      {best['trainable_params']:,} ({best['param_compression']:.1f}x menos)")
print(f"    Recovery:    {best['recovery']:+.2f}%")
print(f"    Gap:         {best['gap_to_baseline']:.2f}%")

print(f"\n  📊 Comparação com MNIST MLP (Exp2):")
print(f"    MNIST:  97.56% acc, 5,770 params, 40.8x compressão (K=128, B=16)")
print(f"    CIFAR:  {best['post_train_accuracy']:.2f}% acc, "
      f"{best['trainable_params']:,} params, "
      f"{best['param_compression']:.1f}x compressão "
      f"(K={best['K']}, B={best['block_size']})")

pct = best['post_train_accuracy'] / baseline_acc * 100
if pct >= 98:
    print(f"\n  ✅ CODEBOOK LEARNING FUNCIONA EM CNN! ({pct:.1f}% do baseline)")
    print(f"     A TESE ESCALA PARA ALÉM DE MNIST MLP.")
elif pct >= 95:
    print(f"\n  ✅ Codebook learning funciona parcialmente ({pct:.1f}% do baseline)")
elif pct >= 90:
    print(f"\n  ⚠️ Recovery moderado ({pct:.1f}% do baseline)")
else:
    print(f"\n  ❌ Recovery insuficiente ({pct:.1f}% do baseline)")

# ── Todas as configs ──
print(f"\n  {'K':>5} {'B':>3} {'Pre%':>7} {'Post%':>7} {'Best%':>7} {'Params':>8} {'Comp':>6} {'Recovery':>10}")
print("  " + "-" * 65)
for e in learning_results["experiments"]:
    print(f"  {e['K']:>5} {e['block_size']:>3} {e['pre_train_accuracy']:>7.2f} "
          f"{e['post_train_accuracy']:>7.2f} {e['best_epoch_accuracy']:>7.2f} "
          f"{e['trainable_params']:>8,} {e['param_compression']:>5.1f}x {e['recovery']:>+9.2f}%")

# %% ════════════════════════════════════════════════════════════
# CÉLULA 7: Download dos Resultados
# ════════════════════════════════════════════════════════════
from google.colab import files
files.download(f"{DATA_DIR}/exp3_quantize_results.json")
files.download(f"{DATA_DIR}/exp3_learning_results.json")
print("✅ Arquivos baixados! Cole-os em pesquisas/tensor-vivo/dados/")
