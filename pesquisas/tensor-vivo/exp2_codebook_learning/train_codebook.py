"""
Tensor-Vivo — Experimento 2: Codebook Learning

A pergunta final e mais importante:
  Se CONGELARMOS os índices (qual centróide cada bloco de pesos usa)
  e TREINARMOS APENAS os valores do codebook, o modelo aprende?

Se sim → o Codebook do Crompressor é um espaço de aprendizado viável.
         Equivale a um "LoRA no espaço comprimido".

Arquitetura:
  - Cada nn.Linear é substituído por CodebookLinear
  - CodebookLinear tem:
    - codebook: Tensor (K, block_size) [TREINÁVEL]
    - indices:  Tensor (N_blocks,)     [CONGELADO]
  - Forward: weight = codebook[indices].reshape(out, in)
  - Backward: gradiente flui para codebook via Straight-Through
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
MODEL_PATH = os.path.join(DATA_DIR, "mnist_mlp.pt")
RESULTS_PATH = os.path.join(DATA_DIR, "exp2_results.json")

sys.path.insert(0, os.path.join(ROOT_DIR, "exp0_analise_estrutural"))
from train_mnist import SimpleMLP


class CodebookLinear(nn.Module):
    """
    Linear layer where weights come from a learned codebook.

    Instead of storing W as (out_features × in_features) floats,
    we store:
      - codebook: (K, block_size) trainable centroids
      - indices: (N_blocks,) frozen assignments
      - bias: (out_features,) trainable (normal)

    Forward pass:
      1. Reconstruct W from codebook[indices]
      2. Compute F.linear(x, W, bias)
    """

    def __init__(self, original_linear, K, block_size):
        super().__init__()
        self.out_features = original_linear.out_features
        self.in_features = original_linear.in_features
        self.K = K
        self.block_size = block_size

        # Extract original weights
        w = original_linear.weight.detach().cpu().numpy().flatten()
        n = len(w)

        # Pad if necessary
        self.original_numel = n
        remainder = n % block_size
        if remainder:
            w = np.concatenate([w, np.zeros(block_size - remainder)])
        self.padded_numel = len(w)

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

        # Bias: TRAINABLE (copy from original)
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.detach().clone())
        else:
            self.bias = None

        self.actual_K = actual_K
        self.num_blocks = num_blocks

    def forward(self, x):
        # Reconstruct weight matrix from codebook
        # codebook[indices] → (num_blocks, block_size)
        reconstructed = self.codebook[self.indices]  # differentiable!
        W = reconstructed.reshape(-1)[:self.original_numel]
        W = W.reshape(self.out_features, self.in_features)
        return F.linear(x, W, self.bias)

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"K={self.actual_K}, block={self.block_size}, "
                f"codebook_params={self.actual_K * self.block_size}")


class CodebookMLP(nn.Module):
    """MLP where all Linear layers use CodebookLinear."""

    def __init__(self, original_model, K, block_size):
        super().__init__()
        self.layers = nn.Sequential(
            CodebookLinear(original_model.layers[0], K, block_size),
            nn.ReLU(),
            CodebookLinear(original_model.layers[2], K, block_size),
            nn.ReLU(),
            CodebookLinear(original_model.layers[4], K, block_size),
        )

    def forward(self, x):
        return self.layers(x.view(x.size(0), -1))


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total * 100


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_frozen_params(model):
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


def main():
    print("=" * 60)
    print("  TENSOR-VIVO — Exp2: CODEBOOK LEARNING")
    print("  Pode o modelo aprender treinando APENAS o codebook?")
    print("=" * 60)

    # ── Dados ──
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(os.path.join(DATA_DIR, "mnist"), train=True,
                              download=True, transform=transform)
    test_ds = datasets.MNIST(os.path.join(DATA_DIR, "mnist"), train=False,
                             transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1000)

    # ── Baselines ──
    original = SimpleMLP()
    original.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    baseline_acc = evaluate(original, test_loader)
    baseline_params = count_trainable_params(original)
    print(f"\n📊 Baseline: {baseline_acc:.2f}% accuracy, {baseline_params:,} params")

    # ── Configurações de teste ──
    configs = [
        # (K, block_size) — baseado nos melhores resultados do Exp1
        (128, 16),   # 96.43% acc, 18.5x compressão no Exp1
        (256, 16),   # 96.31% acc, 14.0x compressão
        (512, 16),   # 96.97% acc, 9.4x compressão
        (128, 32),   # 92.55% acc, 17.9x compressão
        (256, 32),   # 96.00% acc, 11.0x compressão
    ]

    results = {
        "baseline_accuracy": baseline_acc,
        "baseline_params": baseline_params,
        "experiments": [],
    }

    for K, block_size in configs:
        print(f"\n{'━'*60}")
        print(f"  K={K}, Block={block_size}")
        print(f"{'━'*60}")

        # ── Construir CodebookMLP ──
        original = SimpleMLP()
        original.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        model = CodebookMLP(original, K, block_size)

        # ── Métricas pré-treino ──
        pre_train_acc = evaluate(model, test_loader)
        trainable = count_trainable_params(model)
        compression = baseline_params / trainable if trainable else 0

        print(f"  Pré-treino accuracy: {pre_train_acc:.2f}%")
        print(f"  Params treináveis:   {trainable:,} (codebook + biases)")
        print(f"  Params originais:    {baseline_params:,}")
        print(f"  Compressão de params: {compression:.1f}x")
        print(f"\n  {'Epoch':>6} {'Acc%':>8} {'Loss':>10} {'Δ vs Pre':>10}")
        print("  " + "-" * 40)

        # ── Training loop: treinar APENAS codebook + biases ──
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        epoch_data = []

        for epoch in range(1, 21):  # 20 epochs
            model.train()
            total_loss = 0
            for x, y in train_loader:
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            acc = evaluate(model, test_loader)
            delta = acc - pre_train_acc

            print(f"  {epoch:>6} {acc:>8.2f} {avg_loss:>10.4f} {delta:>+9.2f}%")

            epoch_data.append({
                "epoch": epoch,
                "accuracy": round(acc, 4),
                "loss": round(avg_loss, 6),
                "delta_vs_pretrain": round(delta, 4),
            })

        final_acc = epoch_data[-1]["accuracy"]
        recovery = final_acc - pre_train_acc
        gap_to_baseline = baseline_acc - final_acc

        print(f"\n  📊 Resumo K={K}, Block={block_size}:")
        print(f"     Pré-treino:  {pre_train_acc:.2f}%")
        print(f"     Pós-treino:  {final_acc:.2f}%")
        print(f"     Recovery:    {recovery:+.2f}%")
        print(f"     Gap vs base: {gap_to_baseline:.2f}%")
        print(f"     Params:      {trainable:,} ({compression:.1f}x menos)")

        results["experiments"].append({
            "K": K,
            "block_size": block_size,
            "pre_train_accuracy": round(pre_train_acc, 4),
            "post_train_accuracy": final_acc,
            "recovery": round(recovery, 4),
            "gap_to_baseline": round(gap_to_baseline, 4),
            "trainable_params": trainable,
            "param_compression": round(compression, 2),
            "epochs": epoch_data,
        })

    # ── Salvar ──
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Resultados salvos em: {RESULTS_PATH}")

    # ── Veredicto ──
    print(f"\n{'='*60}")
    print("  VEREDICTO FINAL")
    print(f"{'='*60}")

    best = max(results["experiments"], key=lambda x: x["post_train_accuracy"])
    print(f"\n  Melhor resultado:")
    print(f"    K={best['K']}, Block={best['block_size']}")
    print(f"    Accuracy final: {best['post_train_accuracy']:.2f}%")
    print(f"    Recovery:       {best['recovery']:+.2f}%")
    print(f"    Params:         {best['trainable_params']:,} ({best['param_compression']:.1f}x menor)")
    print(f"    Baseline:       {baseline_acc:.2f}%")

    if best['post_train_accuracy'] >= baseline_acc * 0.98:
        print(f"\n  ✅ O CODEBOOK É UM ESPAÇO DE APRENDIZADO VIÁVEL!")
        print(f"     O modelo recuperou para ≥98% do baseline treinando")
        print(f"     APENAS {best['trainable_params']:,} params do codebook.")
        print(f"     ISSO É O 'LoRA DO CROMPRESSOR'.")
    elif best['post_train_accuracy'] >= baseline_acc * 0.95:
        print(f"\n  ✅ O codebook aprende significativamente!")
        print(f"     Recuperação parcial mas expressiva.")
    else:
        print(f"\n  ⚠️ O codebook aprendeu, mas não recuperou accuracy suficiente.")
        print(f"     Possíveis melhorias: mais epochs, learning rate schedule,")
        print(f"     ou K maior.")


if __name__ == "__main__":
    main()
