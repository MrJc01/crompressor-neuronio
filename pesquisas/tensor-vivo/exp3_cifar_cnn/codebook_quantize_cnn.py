"""
Tensor-Vivo — Experimento 3, Parte 2: Codebook Quantization em CNN

Adapta o codebook_quantize.py do Exp1 para a CNN CIFAR-10.
Usa a estratégia Flatten+Chunk (Opção C): achatar todos os pesos
de cada camada e dividir em blocos de block_size.

Grid de teste:
  K ∈ {32, 64, 128, 256, 512}
  block_size ∈ {8, 16, 32}
  = 15 combinações
"""

import os
import sys
import json
import time
import numpy as np
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans
from torchvision import datasets, transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "dados")
MODEL_PATH = os.path.join(DATA_DIR, "cifar_cnn.pt")
RESULTS_PATH = os.path.join(DATA_DIR, "exp3_quantize_results.json")

sys.path.insert(0, SCRIPT_DIR)
from train_cifar_cnn import SimpleCNN, evaluate


def quantize_layer_weights(weight_tensor, K, block_size):
    """Quantiza pesos de uma camada usando K-Means (flatten+chunk)."""
    w = weight_tensor.detach().cpu().numpy().flatten()
    n = len(w)

    remainder = n % block_size
    if remainder != 0:
        w = np.concatenate([w, np.zeros(block_size - remainder)])

    blocks = w.reshape(-1, block_size)
    num_blocks = blocks.shape[0]
    actual_K = min(K, num_blocks)

    if num_blocks > 500:
        kmeans = MiniBatchKMeans(n_clusters=actual_K, random_state=42,
                                batch_size=min(500, num_blocks))
    else:
        kmeans = KMeans(n_clusters=actual_K, random_state=42, n_init=3)

    kmeans.fit(blocks)
    codebook = kmeans.cluster_centers_
    indices = kmeans.labels_

    quantized_flat = codebook[indices].flatten()[:n]
    original_flat = weight_tensor.detach().cpu().numpy().flatten()
    mse = float(np.mean((original_flat - quantized_flat) ** 2))

    original_bytes = n * 4
    compressed_bytes = actual_K * block_size * 4 + num_blocks * 2
    ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else 0
    utilization = len(np.unique(indices)) / actual_K * 100

    return {
        "quantized_flat": quantized_flat,
        "K": actual_K, "blocks": num_blocks, "block_size": block_size,
        "mse": mse, "original_bytes": original_bytes,
        "compressed_bytes": compressed_bytes, "ratio": round(ratio, 2),
        "codebook_utilization": round(utilization, 1),
    }


def main():
    print("=" * 70)
    print("  TENSOR-VIVO — Exp3 Parte 2: Codebook Quantization em CNN CIFAR-10")
    print("=" * 70)

    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Modelo não encontrado: {MODEL_PATH}")
        sys.exit(1)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_ds = datasets.CIFAR10(os.path.join(DATA_DIR, "cifar10"), train=False,
                                transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=500, num_workers=2)

    model = SimpleCNN()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    baseline_acc = evaluate(model, test_loader)
    baseline_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Baseline: {baseline_acc:.2f}% accuracy, {baseline_params:,} params")

    K_values = [32, 64, 128, 256, 512]
    block_sizes = [8, 16, 32]

    results = {
        "baseline_accuracy": baseline_acc,
        "baseline_params": baseline_params,
        "experiments": [],
    }

    print(f"\n{'K':>6} {'Block':>6} {'Acc%':>8} {'Loss':>8} {'Ratio':>8} "
          f"{'Conv MSE':>12} {'FC MSE':>12} {'Util%':>7} {'Time':>7}")
    print("-" * 90)

    for block_size in block_sizes:
        for K in K_values:
            t0 = time.time()
            model = SimpleCNN()
            model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

            total_original = total_compressed = 0
            conv_mse_sum = conv_count = fc_mse_sum = fc_count = 0
            layer_details = []
            total_util = util_count = 0

            for name, param in model.named_parameters():
                if "weight" not in name:
                    continue

                info = quantize_layer_weights(param, K, block_size)
                layer_type = "Conv2d" if param.dim() == 4 else "Linear"

                q_tensor = torch.tensor(
                    info["quantized_flat"][:param.numel()], dtype=torch.float32
                ).reshape(param.shape)
                with torch.no_grad():
                    param.copy_(q_tensor)

                total_original += info["original_bytes"]
                total_compressed += info["compressed_bytes"]
                total_util += info["codebook_utilization"]
                util_count += 1

                if layer_type == "Conv2d":
                    conv_mse_sum += info["mse"]; conv_count += 1
                else:
                    fc_mse_sum += info["mse"]; fc_count += 1

                layer_details.append({
                    "name": name, "type": layer_type,
                    "shape": list(param.shape),
                    "K_actual": info["K"], "blocks": info["blocks"],
                    "mse": round(info["mse"], 8),
                    "ratio": info["ratio"],
                    "utilization": info["codebook_utilization"],
                })

            acc = evaluate(model, test_loader)
            elapsed = time.time() - t0
            acc_loss = baseline_acc - acc
            global_ratio = total_original / total_compressed if total_compressed else 0
            conv_mse = conv_mse_sum / conv_count if conv_count else 0
            fc_mse = fc_mse_sum / fc_count if fc_count else 0
            avg_util = total_util / util_count if util_count else 0

            print(f"{K:>6} {block_size:>6} {acc:>8.2f} {acc_loss:>+7.2f}% {global_ratio:>7.1f}x "
                  f"{conv_mse:>12.6f} {fc_mse:>12.6f} {avg_util:>6.1f}% {elapsed:>6.1f}s")

            results["experiments"].append({
                "K": K, "block_size": block_size,
                "accuracy": round(acc, 4),
                "accuracy_loss": round(acc_loss, 4),
                "compression_ratio": round(global_ratio, 2),
                "conv_mse": round(conv_mse, 8),
                "fc_mse": round(fc_mse, 8),
                "avg_codebook_utilization": round(avg_util, 1),
                "original_bytes": total_original,
                "compressed_bytes": total_compressed,
                "time_seconds": round(elapsed, 2),
                "layer_details": layer_details,
            })

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Resultados salvos em: {RESULTS_PATH}")

    # ── Análise ──
    print(f"\n{'='*70}")
    print("  ANÁLISE DOS RESULTADOS")
    print(f"{'='*70}")

    sorted_by_acc = sorted(results["experiments"], key=lambda x: x["accuracy"], reverse=True)
    print(f"\n  Top 5 por Accuracy:")
    for i, exp in enumerate(sorted_by_acc[:5]):
        print(f"    {i+1}. K={exp['K']:>4} Block={exp['block_size']:>3} → "
              f"{exp['accuracy']:.2f}% (ratio={exp['compression_ratio']:.1f}x)")

    print(f"\n  Análise Conv2d vs Linear (MSE médio):")
    for bs in block_sizes:
        exps_bs = [e for e in results["experiments"] if e["block_size"] == bs]
        avg_conv = np.mean([e["conv_mse"] for e in exps_bs])
        avg_fc = np.mean([e["fc_mse"] for e in exps_bs])
        winner = "Conv2d" if avg_conv < avg_fc else "Linear"
        print(f"    Block={bs:>3}: Conv MSE={avg_conv:.6f}, FC MSE={avg_fc:.6f} → {winner} comprime melhor")

    viable = [e for e in results["experiments"] if e["accuracy"] >= baseline_acc * 0.90]
    if viable:
        best = max(viable, key=lambda x: x["compression_ratio"])
        print(f"\n  ✅ MELHOR TRADEOFF (≥90% baseline):")
        print(f"     K={best['K']}, Block={best['block_size']}")
        print(f"     Accuracy: {best['accuracy']:.2f}% (−{best['accuracy_loss']:.2f}%)")
        print(f"     Compressão: {best['compression_ratio']:.1f}x")


if __name__ == "__main__":
    main()
