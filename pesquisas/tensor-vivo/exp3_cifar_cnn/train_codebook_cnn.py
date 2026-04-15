"""
Tensor-Vivo — Experimento 3, Parte 3: Codebook Learning em CNN CIFAR-10

A pergunta central desta fase:
  O Codebook Learning que funcionou em MNIST MLP também funciona em CNN?

Implementa:
  - CodebookConv2d(nn.Module): Conv2d com pesos via codebook
  - CodebookLinear(nn.Module): Linear com pesos via codebook
  - CodebookCNN: CNN completa com todas as camadas usando codebook
  - Training loop: treinar APENAS codebook + biases (indices congelados)

Arquitetura da CNN original (SimpleCNN):
  features: Conv2d(3→32) → ReLU → MaxPool → Conv2d(32→64) → ReLU → MaxPool
  classifier: Linear(4096→256) → ReLU → Linear(256→10)

Se funcionar → Tese confirmada em escala intermediária (CNN + dados visuais).
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import MiniBatchKMeans, KMeans
from torchvision import datasets, transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "dados")
MODEL_PATH = os.path.join(DATA_DIR, "cifar_cnn.pt")
RESULTS_PATH = os.path.join(DATA_DIR, "exp3_learning_results.json")

sys.path.insert(0, SCRIPT_DIR)
from train_cifar_cnn import SimpleCNN, evaluate


class CodebookConv2d(nn.Module):
    """
    Conv2d where weights come from a learned codebook.

    Instead of storing W as (out_ch, in_ch, kH, kW) floats,
    we store:
      - codebook: (K, block_size) trainable centroids
      - indices: (N_blocks,) frozen assignments
      - bias: (out_ch,) trainable (normal)

    Forward:
      1. Reconstruct W from codebook[indices]
      2. Reshape to (out_ch, in_ch, kH, kW)
      3. F.conv2d(x, W, bias, stride, padding)
    """

    def __init__(self, original_conv2d, K, block_size):
        super().__init__()
        self.out_channels = original_conv2d.out_channels
        self.in_channels = original_conv2d.in_channels
        self.kernel_size = original_conv2d.kernel_size
        self.stride = original_conv2d.stride
        self.padding = original_conv2d.padding
        self.K = K
        self.block_size = block_size

        # Extract and flatten original weights
        w = original_conv2d.weight.detach().cpu().numpy().flatten()
        n = len(w)
        self.original_numel = n
        self.weight_shape = original_conv2d.weight.shape

        # Pad if necessary
        remainder = n % block_size
        if remainder:
            w = np.concatenate([w, np.zeros(block_size - remainder)])

        blocks = w.reshape(-1, block_size)
        num_blocks = blocks.shape[0]
        actual_K = min(K, num_blocks)

        # K-Means clustering
        if num_blocks > 500:
            km = MiniBatchKMeans(n_clusters=actual_K, random_state=42,
                                batch_size=min(500, num_blocks))
        else:
            km = KMeans(n_clusters=actual_K, random_state=42, n_init=3)
        km.fit(blocks)

        # Codebook: TRAINABLE
        self.codebook = nn.Parameter(
            torch.tensor(km.cluster_centers_, dtype=torch.float32)
        )

        # Indices: FROZEN
        self.register_buffer(
            'indices', torch.tensor(km.labels_, dtype=torch.long)
        )

        # Bias: TRAINABLE
        if original_conv2d.bias is not None:
            self.bias = nn.Parameter(original_conv2d.bias.detach().clone())
        else:
            self.bias = None

        self.actual_K = actual_K
        self.num_blocks = num_blocks

    def forward(self, x):
        reconstructed = self.codebook[self.indices]
        W = reconstructed.reshape(-1)[:self.original_numel]
        W = W.reshape(self.weight_shape)
        return F.conv2d(x, W, self.bias, self.stride, self.padding)

    def extra_repr(self):
        return (f"in_ch={self.in_channels}, out_ch={self.out_channels}, "
                f"kernel={self.kernel_size}, K={self.actual_K}, "
                f"block={self.block_size}, "
                f"codebook_params={self.actual_K * self.block_size}")


class CodebookLinear(nn.Module):
    """
    Linear layer where weights come from a learned codebook.
    (Same logic as exp2/train_codebook.py)
    """

    def __init__(self, original_linear, K, block_size):
        super().__init__()
        self.out_features = original_linear.out_features
        self.in_features = original_linear.in_features
        self.K = K
        self.block_size = block_size

        w = original_linear.weight.detach().cpu().numpy().flatten()
        n = len(w)
        self.original_numel = n

        remainder = n % block_size
        if remainder:
            w = np.concatenate([w, np.zeros(block_size - remainder)])

        blocks = w.reshape(-1, block_size)
        num_blocks = blocks.shape[0]
        actual_K = min(K, num_blocks)

        if num_blocks > 500:
            km = MiniBatchKMeans(n_clusters=actual_K, random_state=42,
                                batch_size=min(500, num_blocks))
        else:
            km = KMeans(n_clusters=actual_K, random_state=42, n_init=3)
        km.fit(blocks)

        self.codebook = nn.Parameter(
            torch.tensor(km.cluster_centers_, dtype=torch.float32)
        )
        self.register_buffer(
            'indices', torch.tensor(km.labels_, dtype=torch.long)
        )

        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.detach().clone())
        else:
            self.bias = None

        self.actual_K = actual_K
        self.num_blocks = num_blocks

    def forward(self, x):
        reconstructed = self.codebook[self.indices]
        W = reconstructed.reshape(-1)[:self.original_numel]
        W = W.reshape(self.out_features, self.in_features)
        return F.linear(x, W, self.bias)

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"K={self.actual_K}, block={self.block_size}, "
                f"codebook_params={self.actual_K * self.block_size}")


class CodebookCNN(nn.Module):
    """
    CNN where ALL weight layers (Conv2d + Linear) use codebook representation.

    Maps the original SimpleCNN architecture:
      features[0] = Conv2d(3,32)  → CodebookConv2d
      features[1] = ReLU          (unchanged)
      features[2] = MaxPool2d      (unchanged)
      features[3] = Conv2d(32,64) → CodebookConv2d
      features[4] = ReLU          (unchanged)
      features[5] = MaxPool2d      (unchanged)
      classifier[0] = Linear(4096,256) → CodebookLinear
      classifier[1] = ReLU             (unchanged)
      classifier[2] = Linear(256,10)   → CodebookLinear
    """

    def __init__(self, original_model, K, block_size):
        super().__init__()
        self.features = nn.Sequential(
            CodebookConv2d(original_model.features[0], K, block_size),
            nn.ReLU(),
            nn.MaxPool2d(2),
            CodebookConv2d(original_model.features[3], K, block_size),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            CodebookLinear(original_model.classifier[0], K, block_size),
            nn.ReLU(),
            CodebookLinear(original_model.classifier[2], K, block_size),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_layer_info(model):
    """Extract per-layer codebook info for analysis."""
    info = []
    for name, module in model.named_modules():
        if isinstance(module, (CodebookConv2d, CodebookLinear)):
            layer_type = "Conv2d" if isinstance(module, CodebookConv2d) else "Linear"
            info.append({
                "name": name,
                "type": layer_type,
                "K_actual": module.actual_K,
                "block_size": module.block_size,
                "num_blocks": module.num_blocks,
                "codebook_params": module.actual_K * module.block_size,
                "bias_params": module.bias.numel() if module.bias is not None else 0,
            })
    return info


def main():
    print("=" * 70)
    print("  TENSOR-VIVO — Exp3 Parte 3: CODEBOOK LEARNING em CNN CIFAR-10")
    print("  Pode o modelo CNN aprender treinando APENAS o codebook?")
    print("=" * 70)

    # ── Verificar modelo ──
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Modelo não encontrado em {MODEL_PATH}")
        print("   Rode primeiro: python train_cifar_cnn.py")
        sys.exit(1)

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

    train_ds = datasets.CIFAR10(os.path.join(DATA_DIR, "cifar10"), train=True,
                                 download=True, transform=train_transform)
    test_ds = datasets.CIFAR10(os.path.join(DATA_DIR, "cifar10"), train=False,
                                transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128,
                                                shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=500,
                                               num_workers=2)

    # ── Baseline ──
    original = SimpleCNN()
    original.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    baseline_acc = evaluate(original, test_loader)
    baseline_params = sum(p.numel() for p in original.parameters())
    print(f"\n📊 Baseline CNN: {baseline_acc:.2f}% accuracy, {baseline_params:,} params")

    # ── Configurações de teste ──
    configs = [
        (128, 8),    # Alta granularidade, K médio
        (256, 8),    # Alta granularidade, K alto
        (128, 16),   # Comparação direta com MNIST Exp2
        (256, 16),   # Match com melhor MNIST config
        (256, 32),   # Máxima compressão viável
    ]

    results = {
        "baseline_accuracy": baseline_acc,
        "baseline_params": baseline_params,
        "model": "SimpleCNN_CIFAR10",
        "experiments": [],
    }

    for K, block_size in configs:
        print(f"\n{'━'*70}")
        print(f"  K={K}, Block={block_size}")
        print(f"{'━'*70}")

        # ── Construir CodebookCNN ──
        original = SimpleCNN()
        original.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        model = CodebookCNN(original, K, block_size)

        # ── Métricas pré-treino ──
        pre_train_acc = evaluate(model, test_loader)
        trainable = count_trainable_params(model)
        compression = baseline_params / trainable if trainable else 0
        layer_info = get_layer_info(model)

        print(f"  Pré-treino accuracy: {pre_train_acc:.2f}%")
        print(f"  Params treináveis:   {trainable:,} (codebook + biases)")
        print(f"  Params originais:    {baseline_params:,}")
        print(f"  Compressão de params: {compression:.1f}x")

        # Per-layer breakdown
        print(f"\n  {'Camada':<25} {'Tipo':<8} {'K':>5} {'Blocks':>8} {'CB Params':>10}")
        print("  " + "-" * 60)
        for li in layer_info:
            print(f"  {li['name']:<25} {li['type']:<8} {li['K_actual']:>5} "
                  f"{li['num_blocks']:>8} {li['codebook_params']:>10,}")

        print(f"\n  {'Epoch':>6} {'Acc%':>8} {'Loss':>10} {'Δ vs Pre':>10} {'Δ vs Base':>10}")
        print("  " + "-" * 50)

        # ── Training loop ──
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        criterion = nn.CrossEntropyLoss()
        epoch_data = []

        t0 = time.time()
        for epoch in range(1, 21):
            model.train()
            total_loss = 0
            for x, y in train_loader:
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / len(train_loader)
            acc = evaluate(model, test_loader)
            delta_pre = acc - pre_train_acc
            delta_base = acc - baseline_acc

            print(f"  {epoch:>6} {acc:>8.2f} {avg_loss:>10.4f} {delta_pre:>+9.2f}% {delta_base:>+9.2f}%")

            epoch_data.append({
                "epoch": epoch,
                "accuracy": round(acc, 4),
                "loss": round(avg_loss, 6),
                "delta_vs_pretrain": round(delta_pre, 4),
                "delta_vs_baseline": round(delta_base, 4),
            })

        elapsed = time.time() - t0
        final_acc = epoch_data[-1]["accuracy"]
        best_epoch_acc = max(e["accuracy"] for e in epoch_data)
        recovery = final_acc - pre_train_acc
        gap_to_baseline = baseline_acc - final_acc

        print(f"\n  📊 Resumo K={K}, Block={block_size}:")
        print(f"     Pré-treino:    {pre_train_acc:.2f}%")
        print(f"     Pós-treino:    {final_acc:.2f}%")
        print(f"     Melhor epoch:  {best_epoch_acc:.2f}%")
        print(f"     Recovery:      {recovery:+.2f}%")
        print(f"     Gap vs base:   {gap_to_baseline:.2f}%")
        print(f"     Params:        {trainable:,} ({compression:.1f}x menos)")
        print(f"     Tempo treino:  {elapsed:.1f}s")

        results["experiments"].append({
            "K": K,
            "block_size": block_size,
            "pre_train_accuracy": round(pre_train_acc, 4),
            "post_train_accuracy": final_acc,
            "best_epoch_accuracy": best_epoch_acc,
            "recovery": round(recovery, 4),
            "gap_to_baseline": round(gap_to_baseline, 4),
            "trainable_params": trainable,
            "param_compression": round(compression, 2),
            "training_time_seconds": round(elapsed, 1),
            "layer_info": layer_info,
            "epochs": epoch_data,
        })

    # ── Salvar ──
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Resultados salvos em: {RESULTS_PATH}")

    # ── Veredicto ──
    print(f"\n{'='*70}")
    print("  VEREDICTO — CODEBOOK LEARNING EM CNN")
    print(f"{'='*70}")

    best = max(results["experiments"], key=lambda x: x["post_train_accuracy"])
    most_compressed = max(
        [e for e in results["experiments"] if e["post_train_accuracy"] >= baseline_acc * 0.95],
        key=lambda x: x["param_compression"],
        default=None
    )

    print(f"\n  Melhor accuracy:")
    print(f"    K={best['K']}, Block={best['block_size']}")
    print(f"    Accuracy: {best['post_train_accuracy']:.2f}% "
          f"(baseline: {baseline_acc:.2f}%)")
    print(f"    Params: {best['trainable_params']:,} ({best['param_compression']:.1f}x menor)")

    if most_compressed:
        print(f"\n  Melhor tradeoff (≥95% baseline):")
        print(f"    K={most_compressed['K']}, Block={most_compressed['block_size']}")
        print(f"    Accuracy: {most_compressed['post_train_accuracy']:.2f}%")
        print(f"    Compressão: {most_compressed['param_compression']:.1f}x")

    # ── Comparação com MNIST ──
    print(f"\n  📊 Comparação MNIST MLP vs CIFAR-10 CNN:")
    print(f"    MNIST:   97.56% acc, 5,770 params, 40.8x compressão (K=128, B=16)")
    print(f"    CIFAR:   {best['post_train_accuracy']:.2f}% acc, "
          f"{best['trainable_params']:,} params, "
          f"{best['param_compression']:.1f}x compressão "
          f"(K={best['K']}, B={best['block_size']})")

    if best['post_train_accuracy'] >= baseline_acc * 0.98:
        print(f"\n  ✅ CODEBOOK LEARNING FUNCIONA EM CNN!")
        print(f"     O modelo CNN recuperou ≥98% do baseline treinando")
        print(f"     APENAS {best['trainable_params']:,} params do codebook.")
        print(f"     A TESE ESCALA PARA ALÉM DE MNIST MLP.")
    elif best['post_train_accuracy'] >= baseline_acc * 0.95:
        print(f"\n  ✅ Codebook learning funciona parcialmente em CNN.")
        print(f"     Recovery significativo mas com gap residual.")
    elif best['post_train_accuracy'] >= baseline_acc * 0.90:
        print(f"\n  ⚠️ Codebook learning tem recovery moderado em CNN.")
        print(f"     Conv2d pode precisar de tratamento diferente.")
    else:
        print(f"\n  ❌ Codebook learning tem dificuldade em CNN.")
        print(f"     Investigar: lr, K maior, block_size diferente, ou")
        print(f"     quantização seletiva (só Linear, não Conv2d).")


if __name__ == "__main__":
    main()
