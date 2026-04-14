"""
Tensor-Vivo — Experimento 1: Roundtrip com Codebook Quantization

A Fase 0 mostrou que CDC hash exato não encontra dedup em pesos float32 —
cada neurônio é numericamente único. Mas isso NÃO invalida a tese.

A pergunta real é: se agruparmos neurônios "próximos" em clusters (codebook)
e substituirmos cada peso pelo centróide do cluster mais próximo, o modelo
ainda funciona?

Este experimento faz:
1. Divide pesos de cada camada em blocos de N floats
2. K-means para agrupar blocos similares em K clusters
3. Substitui cada bloco pelo centróide mais próximo (quantização)
4. Inferência com pesos quantizados → mede accuracy
5. Varia K: 8, 16, 32, 64, 128, 256, 512, 1024
6. Calcula compression ratio e plota curvas
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans, MiniBatchKMeans
from torchvision import datasets, transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "dados")
MODEL_PATH = os.path.join(DATA_DIR, "mnist_mlp.pt")
RESULTS_PATH = os.path.join(DATA_DIR, "exp1_results.json")

sys.path.insert(0, os.path.join(ROOT_DIR, "exp0_analise_estrutural"))
from train_mnist import SimpleMLP


def evaluate(model, test_loader):
    """Avalia accuracy do modelo."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total * 100


def quantize_layer_weights(weight_tensor, K, block_size):
    """
    Quantiza os pesos de uma camada usando K-Means.

    weight_tensor: tensor (out_features, in_features)
    K: número de centróides no codebook
    block_size: tamanho de cada bloco de floats para clustering

    Retorna: (quantized_weights, codebook, indices, info)
    """
    w = weight_tensor.detach().cpu().numpy().flatten()
    n = len(w)

    # Dividir em blocos de block_size floats
    # Se não for divisível, pad com zeros
    remainder = n % block_size
    if remainder != 0:
        w = np.concatenate([w, np.zeros(block_size - remainder)])

    blocks = w.reshape(-1, block_size)
    num_blocks = blocks.shape[0]

    # Limitar K ao número de blocos
    actual_K = min(K, num_blocks)

    # K-Means (usar MiniBatch para velocidade em datasets grandes)
    if num_blocks > 1000:
        kmeans = MiniBatchKMeans(n_clusters=actual_K, random_state=42,
                                batch_size=min(1000, num_blocks))
    else:
        kmeans = KMeans(n_clusters=actual_K, random_state=42, n_init=3)

    kmeans.fit(blocks)
    indices = kmeans.labels_
    codebook = kmeans.cluster_centers_  # (K, block_size)

    # Reconstruir pesos quantizados
    quantized_blocks = codebook[indices]  # (num_blocks, block_size)
    quantized_flat = quantized_blocks.flatten()[:n]  # remover padding

    # Métricas
    mse = np.mean((weight_tensor.detach().cpu().numpy().flatten() - quantized_flat) ** 2)
    original_bytes = n * 4  # float32
    codebook_bytes = actual_K * block_size * 4  # float32 centróides
    index_bytes = num_blocks * 2  # uint16 índices
    compressed_bytes = codebook_bytes + index_bytes
    ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else 0

    return {
        "quantized_flat": quantized_flat,
        "codebook": codebook,
        "indices": indices,
        "K": actual_K,
        "blocks": num_blocks,
        "block_size": block_size,
        "mse": float(mse),
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_bytes,
        "ratio": round(ratio, 2),
    }


def main():
    print("=" * 60)
    print("  TENSOR-VIVO — Exp1: Roundtrip com Codebook")
    print("=" * 60)

    # ── Dados de teste ──
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_ds = datasets.MNIST(os.path.join(DATA_DIR, "mnist"), train=False,
                             transform=transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1000)

    # ── Modelo original ──
    model = SimpleMLP()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    baseline_acc = evaluate(model, test_loader)
    print(f"\n📊 Accuracy Baseline (pesos originais): {baseline_acc:.2f}%")

    # ── Configurações de teste ──
    K_values = [4, 8, 16, 32, 64, 128, 256, 512]
    block_sizes = [16, 32, 64, 128]

    results = {
        "baseline_accuracy": baseline_acc,
        "experiments": [],
    }

    print(f"\n{'K':>6} {'Block':>6} {'Acc%':>8} {'Loss':>8} {'Ratio':>8} {'MSE':>12} {'Time':>8}")
    print("-" * 70)

    for block_size in block_sizes:
        for K in K_values:
            t0 = time.time()

            # Carregar modelo fresco
            model = SimpleMLP()
            model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

            total_original = 0
            total_compressed = 0
            total_mse = 0
            num_layers = 0

            # Quantizar cada camada de pesos
            for name, param in model.named_parameters():
                if "weight" not in name:
                    continue

                info = quantize_layer_weights(param, K, block_size)

                # Injetar pesos quantizados
                shape = param.shape
                q_tensor = torch.tensor(
                    info["quantized_flat"][:param.numel()],
                    dtype=torch.float32
                ).reshape(shape)

                with torch.no_grad():
                    param.copy_(q_tensor)

                total_original += info["original_bytes"]
                total_compressed += info["compressed_bytes"]
                total_mse += info["mse"]
                num_layers += 1

            # Avaliar
            acc = evaluate(model, test_loader)
            elapsed = time.time() - t0
            acc_loss = baseline_acc - acc
            global_ratio = total_original / total_compressed if total_compressed else 0
            avg_mse = total_mse / num_layers

            print(f"{K:>6} {block_size:>6} {acc:>8.2f} {acc_loss:>+7.2f}% {global_ratio:>7.1f}x {avg_mse:>12.6f} {elapsed:>7.1f}s")

            results["experiments"].append({
                "K": K,
                "block_size": block_size,
                "accuracy": round(acc, 4),
                "accuracy_loss": round(acc_loss, 4),
                "compression_ratio": round(global_ratio, 2),
                "avg_mse": round(avg_mse, 8),
                "original_bytes": total_original,
                "compressed_bytes": total_compressed,
                "time_seconds": round(elapsed, 2),
            })

    # ── Salvar ──
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Resultados salvos em: {RESULTS_PATH}")

    # ── Análise ──
    print(f"\n{'='*60}")
    print("  ANÁLISE DOS RESULTADOS")
    print(f"{'='*60}")

    # Encontrar melhor tradeoff
    best = None
    for exp in results["experiments"]:
        if exp["accuracy"] >= baseline_acc * 0.95:  # Mantém 95% da accuracy
            if best is None or exp["compression_ratio"] > best["compression_ratio"]:
                best = exp

    if best:
        print(f"\n  ✅ MELHOR TRADEOFF (≥95% accuracy):")
        print(f"     K={best['K']}, Block={best['block_size']}")
        print(f"     Accuracy: {best['accuracy']:.2f}% (−{best['accuracy_loss']:.2f}%)")
        print(f"     Compressão: {best['compression_ratio']:.1f}x")
        print(f"     MSE: {best['avg_mse']:.8f}")
        print(f"\n  → ISSO PROVA que os pesos podem ser representados por CODEBOOK")
        print(f"    com perda aceitável de accuracy.")
    else:
        print(f"\n  ⚠️ Nenhuma configuração manteve ≥95% accuracy.")
        print(f"     Isso sugere que o modelo é muito sensível à quantização")
        print(f"     ou os parâmetros de K/block_size precisam de ajuste.")

    # Mostrar top 5 por accuracy
    sorted_by_acc = sorted(results["experiments"], key=lambda x: x["accuracy"], reverse=True)
    print(f"\n  Top 5 por Accuracy:")
    for i, exp in enumerate(sorted_by_acc[:5]):
        print(f"    {i+1}. K={exp['K']:>4} Block={exp['block_size']:>3} → {exp['accuracy']:.2f}% (ratio={exp['compression_ratio']:.1f}x)")

    # Mostrar top 5 por compressão (com accuracy > 90%)
    viable = [e for e in results["experiments"] if e["accuracy"] >= baseline_acc * 0.90]
    sorted_by_ratio = sorted(viable, key=lambda x: x["compression_ratio"], reverse=True)
    print(f"\n  Top 5 por Compressão (≥90% accuracy):")
    for i, exp in enumerate(sorted_by_ratio[:5]):
        print(f"    {i+1}. K={exp['K']:>4} Block={exp['block_size']:>3} → {exp['compression_ratio']:.1f}x (acc={exp['accuracy']:.2f}%)")


if __name__ == "__main__":
    main()
