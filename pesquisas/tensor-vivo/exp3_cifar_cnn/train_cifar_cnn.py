"""
Tensor-Vivo — Experimento 3, Parte 1: Treinar CNN no CIFAR-10

Treina uma CNN simples no CIFAR-10 para servir de baseline:
  Conv2d(3→32, 3×3) → ReLU → MaxPool
  Conv2d(32→64, 3×3) → ReLU → MaxPool
  Flatten → Linear(64*8*8→256) → ReLU → Linear(256→10)

Sem BatchNorm para simplificar análise do codebook.
Baseline alcançado: ~80% accuracy.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# ── Paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "dados")
MODEL_PATH = os.path.join(DATA_DIR, "cifar_cnn.pt")
META_PATH = os.path.join(DATA_DIR, "exp3_model_meta.json")

os.makedirs(DATA_DIR, exist_ok=True)


# ── Modelo ──
class SimpleCNN(nn.Module):
    """
    CNN compacta para CIFAR-10.
    Sem BatchNorm para manter a análise do codebook limpa.

    Arquitetura:
      Conv2d(3, 32, 3, pad=1) → ReLU → MaxPool(2)   # 32×16×16
      Conv2d(32, 64, 3, pad=1) → ReLU → MaxPool(2)   # 64×8×8
      Flatten → Linear(4096, 256) → ReLU → Linear(256, 10)
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 3×32×32 → 32×32×32
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → 32×16×16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → 64×16×16
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → 64×8×8
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
    """Avalia accuracy do modelo."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total * 100


def train():
    print("=" * 60)
    print("  TENSOR-VIVO — Exp3: Treinar CNN CIFAR-10")
    print("=" * 60)

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

    # ── Treinar ──
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    weight_params = sum(p.numel() for name, p in model.named_parameters() if "weight" in name)
    bias_params = sum(p.numel() for name, p in model.named_parameters() if "bias" in name)

    print(f"\nArquitetura: {model}")
    print(f"Total de parâmetros: {total_params:,}")
    print(f"  Weight params: {weight_params:,}")
    print(f"  Bias params:   {bias_params:,}")

    best_acc = 0
    epoch_history = []

    for epoch in range(1, 31):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        acc = evaluate(model, test_loader)

        lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch:2d}: loss={avg_loss:.4f}  accuracy={acc:.2f}%  lr={lr:.1e}")

        epoch_history.append({
            "epoch": epoch,
            "loss": round(avg_loss, 4),
            "accuracy": round(acc, 2),
            "lr": lr,
        })

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_PATH)

        if acc >= 85.0 and epoch >= 10:
            print(f"  ✅ Accuracy ≥ 85% alcançada no epoch {epoch}. Parando.")
            break

    # ── Carregar melhor modelo ──
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    final_acc = evaluate(model, test_loader)
    print(f"\n📊 Melhor accuracy (test): {final_acc:.2f}%")
    print(f"💾 Modelo salvo em: {MODEL_PATH}")

    # ── Metadados dos pesos ──
    layer_info = []
    total_bytes = 0
    print(f"\n📊 Anatomia dos Pesos:")
    print(f"  {'Camada':<30} {'Shape':<25} {'Params':>10} {'Bytes':>12} {'Tipo':<8}")
    print("  " + "-" * 90)

    for name, param in model.named_parameters():
        shape = tuple(param.shape)
        n_params = param.numel()
        n_bytes = n_params * 4  # float32
        total_bytes += n_bytes
        layer_type = "Conv2d" if len(shape) == 4 else ("Linear" if len(shape) == 2 else "Bias")
        print(f"  {name:<30} {str(shape):<25} {n_params:>10,} {n_bytes:>12,} {layer_type:<8}")
        layer_info.append({
            "name": name,
            "shape": list(shape),
            "params": n_params,
            "bytes": n_bytes,
            "type": layer_type,
        })

    print("  " + "-" * 90)
    print(f"  {'TOTAL':<30} {'':<25} {total_params:>10,} {total_bytes:>12,}")

    meta = {
        "model": "SimpleCNN_CIFAR10",
        "architecture": "Conv2d(3→32→64) + Linear(4096→256→10)",
        "dataset": "CIFAR-10",
        "accuracy": final_acc,
        "total_params": total_params,
        "total_bytes": total_bytes,
        "epochs_trained": len(epoch_history),
        "layers": layer_info,
        "epoch_history": epoch_history,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n📄 Metadados salvos em: {META_PATH}")

    return model, final_acc


if __name__ == "__main__":
    train()
