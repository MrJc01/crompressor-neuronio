"""
CROM Compressor v3 — Compressão de Pesos REAIS de Rede Neural
=============================================================
Extrai os tensores Float32 do GPT-2 (124M parâmetros) carregado na RAM,
comprime via Vector Quantization (K-Means), e mede o erro de reconstrução.

DADOS 100% REAIS. Nenhum corpus sintético.
"""
import os
import sys
import time
import struct
import math
import mmap
import numpy as np
from sklearn.cluster import MiniBatchKMeans

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
print(f"  03. CROM Compressor v3 | Pesos REAIS do GPT-2 (Vector Quantization)")
print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")

try:
    import torch
    from transformers import AutoModelForCausalLM
except ImportError:
    print(f"{Colors.FAIL}[ERRO] PyTorch/Transformers não encontrados.{Colors.ENDC}")
    sys.exit(1)


def extract_real_weights(model):
    """Extrai tensores reais do GPT-2 e os achata em uma matriz (N, D)."""
    print(f"\n[1] Extraindo pesos REAIS do GPT-2...")
    
    # Pegar as camadas de atenção (onde a compressão importa de verdade)
    all_weights = []
    layer_info = []
    
    for name, param in model.named_parameters():
        if 'attn' in name and 'weight' in name:
            w = param.detach().cpu().numpy().astype(np.float32)
            original_shape = w.shape
            flat = w.reshape(-1)
            all_weights.append(flat)
            layer_info.append((name, original_shape, flat.shape[0]))
    
    print(f"    Camadas de Atenção extraídas: {len(layer_info)}")
    for name, shape, n_params in layer_info[:3]:
        print(f"    - {name}: {shape} ({n_params:,} parâmetros)")
    if len(layer_info) > 3:
        print(f"    - ... e mais {len(layer_info)-3} camadas")
    
    # Concatenar tudo e reshapear em blocos de 64 dimensões
    concatenated = np.concatenate(all_weights)
    total_params = len(concatenated)
    dim = 64
    # Truncar para múltiplo de dim
    usable = (total_params // dim) * dim
    matrix = concatenated[:usable].reshape(-1, dim)
    
    print(f"    Total de parâmetros de atenção: {total_params:,}")
    print(f"    Matriz para compressão: {matrix.shape} (N={matrix.shape[0]:,}, D={dim})")
    
    return matrix, total_params


def compress_vq(matrix, k_clusters=256):
    """Comprime via Vector Quantization K-Means."""
    print(f"\n[2] Treinando Codebook K-Means (K={k_clusters})...")
    t0 = time.perf_counter()
    
    kmeans = MiniBatchKMeans(n_clusters=k_clusters, random_state=42, batch_size=4096, max_iter=100)
    labels = kmeans.fit_predict(matrix)
    centroids = kmeans.cluster_centers_.astype(np.float32)
    
    elapsed = time.perf_counter() - t0
    print(f"    Codebook treinado em {elapsed:.2f}s")
    
    return labels, centroids


def measure_reconstruction(matrix, labels, centroids):
    """Mede a qualidade da reconstrução — o número que importa de verdade."""
    print(f"\n[3] Medindo Erro de Reconstrução...")
    
    # Reconstruir a matriz usando o codebook
    reconstructed = centroids[labels]
    
    # MSE (Mean Squared Error)
    mse = np.mean((matrix - reconstructed) ** 2)
    
    # Cosine Similarity médio (por vetor)
    dot_products = np.sum(matrix * reconstructed, axis=1)
    norms_orig = np.linalg.norm(matrix, axis=1)
    norms_recon = np.linalg.norm(reconstructed, axis=1)
    # Evitar divisão por zero
    mask = (norms_orig > 1e-10) & (norms_recon > 1e-10)
    cosine_sims = dot_products[mask] / (norms_orig[mask] * norms_recon[mask])
    avg_cosine = np.mean(cosine_sims)
    
    # SNR (Signal-to-Noise Ratio)
    signal_power = np.mean(matrix ** 2)
    noise_power = mse
    snr_db = 10 * math.log10(signal_power / (noise_power + 1e-10))
    
    print(f"    MSE (Mean Squared Error)  : {mse:.6f}")
    print(f"    Cosine Similarity Médio   : {avg_cosine:.4f} ({avg_cosine*100:.1f}%)")
    print(f"    SNR (Signal-to-Noise)     : {snr_db:.1f} dB")
    
    return mse, avg_cosine, snr_db


def save_crom_file(filepath, matrix, labels, centroids, k_clusters):
    """Grava o pacote .crom v2."""
    print(f"\n[4] Gravando arquivo .crom v2...")
    
    N, D = matrix.shape
    with open(filepath, 'wb') as f:
        f.write(b'CROM')
        f.write(struct.pack('B', 2))
        f.write(struct.pack('I', N))
        f.write(struct.pack('I', D))
        f.write(struct.pack('I', k_clusters))
        f.write(centroids.tobytes())
        # uint16 para suportar K > 256
        f.write(labels.astype(np.uint16).tobytes())
    
    return os.path.getsize(filepath)


def read_crom_mmap(filepath):
    """Leitura Zero-Copy via Memory-Mapped File."""
    print(f"\n[5] Leitura Zero-Copy (MMap)...")
    t0 = time.perf_counter()
    
    with open(filepath, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        magic = mm[:4]
        version = struct.unpack('B', mm[4:5])[0]
        N = struct.unpack('I', mm[5:9])[0]
        D = struct.unpack('I', mm[9:13])[0]
        K = struct.unpack('I', mm[13:17])[0]
        mm.close()
    
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"    Header lido em {elapsed:.3f}ms (Zero-Copy)")
    print(f"    Verificado: N={N:,}, D={D}, K={K}")


def main():
    # Carregar GPT-2 real
    print(f"\n[0] Carregando GPT-2...")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # Extrair pesos reais
    matrix, total_params = extract_real_weights(model)
    
    # Tamanho bruto em disco
    raw_file = "gpt2_attn_raw.bin"
    with open(raw_file, 'wb') as f:
        f.write(matrix.tobytes())
    raw_size = os.path.getsize(raw_file)
    
    # Comprimir
    k = 2048
    labels, centroids = compress_vq(matrix, k_clusters=k)
    
    # Medir qualidade
    mse, cosine, snr = measure_reconstruction(matrix, labels, centroids)
    
    # Salvar .crom
    crom_file = "gpt2_attn_compressed.crom"
    crom_size = save_crom_file(crom_file, matrix, labels, centroids, k)
    
    # MMap
    read_crom_mmap(crom_file)
    
    # Relatório
    compression = (1.0 - (crom_size / raw_size)) * 100
    
    print(f"\n{Colors.OKGREEN}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}📊 RELATÓRIO DE COMPRESSÃO (DADOS REAIS){Colors.ENDC}")
    print(f"{'='*80}")
    print(f"  Fonte dos Dados      : Pesos de Atenção do GPT-2 (124M params)")
    print(f"  Parâmetros Extraídos : {total_params:,}")
    print(f"  Matriz Comprimida    : {matrix.shape}")
    print(f"  Tamanho Bruto (F32)  : {raw_size/1024/1024:.2f} MB")
    print(f"  Tamanho CROM (VQ)    : {crom_size/1024/1024:.2f} MB")
    print(f"  {Colors.BOLD}Taxa de Compressão     : {compression:.2f}%{Colors.ENDC}")
    print(f"  {Colors.BOLD}Fidelidade (Cosine)    : {cosine*100:.1f}%{Colors.ENDC}")
    print(f"  SNR                  : {snr:.1f} dB")
    print(f"{'='*80}")
    
    # Limpar arquivo bruto
    os.remove(raw_file)


if __name__ == "__main__":
    main()
