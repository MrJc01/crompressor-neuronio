"""
Tensor-Vivo — Experimento 0, Parte 1: Treinar MLP no MNIST

Treina um MLP simples (784 → 256 → 128 → 10) no MNIST,
salva o modelo treinado e reporta accuracy baseline + tamanho dos pesos.
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
MODEL_PATH = os.path.join(DATA_DIR, "mnist_mlp.pt")
META_PATH = os.path.join(DATA_DIR, "exp0_model_meta.json")

os.makedirs(DATA_DIR, exist_ok=True)


# ── Modelo ──
class SimpleMLP(nn.Module):
    """MLP mínimo para MNIST: 3 camadas lineares."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.layers(x.view(x.size(0), -1))


def train():
    print("=" * 60)
    print("  TENSOR-VIVO — Exp0: Treinar MLP MNIST")
    print("=" * 60)

    # ── Dados ──
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(os.path.join(DATA_DIR, "mnist"), train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(os.path.join(DATA_DIR, "mnist"), train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1000)

    # ── Treinar ──
    model = SimpleMLP()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"\nArquitetura: {model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total de parâmetros: {total_params:,}")

    for epoch in range(1, 11):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Avaliar
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                out = model(batch_x)
                preds = out.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        acc = correct / total * 100
        print(f"  Epoch {epoch:2d}: loss={total_loss/len(train_loader):.4f}  accuracy={acc:.2f}%")

        if acc > 97.5:
            print(f"  ✅ Accuracy > 97.5% alcançada no epoch {epoch}. Parando.")
            break

    # ── Salvar ──
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n💾 Modelo salvo em: {MODEL_PATH}")

    # ── Metadados dos pesos ──
    layer_info = []
    total_bytes = 0
    print("\n📊 Anatomia dos Pesos:")
    print(f"  {'Camada':<25} {'Shape':<20} {'Params':>10} {'Bytes':>12}")
    print("  " + "-" * 70)

    for name, param in model.named_parameters():
        shape = tuple(param.shape)
        n_params = param.numel()
        n_bytes = n_params * 4  # float32
        total_bytes += n_bytes
        print(f"  {name:<25} {str(shape):<20} {n_params:>10,} {n_bytes:>12,}")
        layer_info.append({
            "name": name,
            "shape": list(shape),
            "params": n_params,
            "bytes": n_bytes,
        })

    print("  " + "-" * 70)
    print(f"  {'TOTAL':<25} {'':<20} {total_params:>10,} {total_bytes:>12,}")

    meta = {
        "model": "SimpleMLP_784_256_128_10",
        "dataset": "MNIST",
        "accuracy": acc,
        "total_params": total_params,
        "total_bytes": total_bytes,
        "layers": layer_info,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n📄 Metadados salvos em: {META_PATH}")

    return model, acc


if __name__ == "__main__":
    train()
