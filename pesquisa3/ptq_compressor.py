import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import struct
import os
import argparse
import time

try:
    import faiss
except ImportError:
    print("FAISS não encontrado. Para compressão PTQ rápida, instale: pip install faiss-cpu (ou faiss-gpu)")
    print("O script usará PyTorch puro para K-Means se o FAISS falhar, mas será muito mais lento.")
    faiss = None


def kmeans_faiss(x: torch.Tensor, K: int, n_iter: int = 20) -> tuple:
    """Executa K-Means ultrarrápido usando FAISS."""
    x_np = x.detach().cpu().numpy().astype(np.float32)
    d = x_np.shape[1]
    
    # Treina o K-Means
    kmeans = faiss.Kmeans(d, K, niter=n_iter, verbose=False, gpu=faiss.get_num_gpus() > 0 if faiss else False)
    kmeans.train(x_np)
    
    codebook = torch.from_numpy(kmeans.centroids).to(x.device)
    
    # Faz o assignment (lookup do vizinho mais próximo)
    index = faiss.IndexFlatL2(d)
    if faiss.get_num_gpus() > 0:
        index = faiss.index_cpu_to_all_gpus(index)
    index.add(kmeans.centroids)
    
    _, indices = index.search(x_np, 1)
    indices = torch.from_numpy(indices).squeeze(-1).to(x.device)
    
    return codebook, indices


def kmeans_pytorch(x: torch.Tensor, K: int, n_iter: int = 20) -> tuple:
    """Fallback K-Means usando PyTorch."""
    # Inicialização aleatória dos centroides
    N, D = x.shape
    indices = torch.randperm(N, device=x.device)[:K]
    codebook = x[indices].clone()
    
    for _ in range(n_iter):
        # Distâncias: (N, K)
        dists = torch.cdist(x, codebook)
        indices = dists.argmin(dim=1)
        
        # Atualiza centroides
        for k in range(K):
            mask = indices == k
            if mask.any():
                codebook[k] = x[mask].mean(dim=0)
                
    return codebook, indices


def compress_linear(weight: torch.Tensor, K: int = 256, D: int = 8) -> tuple:
    """Comprime uma matriz densa de pesos usando PTQ (Post-Training Quantization)."""
    out_features, in_features = weight.shape
    total_elements = in_features * out_features
    
    # Padding se necessário (geralmente matrizes LLM são múltiplas de 8/64)
    padded_elements = ((total_elements + D - 1) // D) * D
    if padded_elements > total_elements:
        # Preenche com zeros o que faltar
        pad_size = padded_elements - total_elements
        weight_flat = torch.cat([weight.flatten(), torch.zeros(pad_size, device=weight.device)])
    else:
        weight_flat = weight.flatten()
        
    # Reshape para blocos de dimensão D
    blocks = weight_flat.reshape(-1, D)
    
    # Roda K-Means
    if faiss is not None:
        codebook, indices = kmeans_faiss(blocks, K)
    else:
        codebook, indices = kmeans_pytorch(blocks, K)
        
    # Calcula Erro Quadrático Médio (MSE) para log
    reconstructed = codebook[indices]
    mse = torch.nn.functional.mse_loss(blocks, reconstructed).item()
    
    return codebook.half(), indices.to(torch.uint16), mse


def save_cromv3_sota(model_name: str, config_dict: dict, tensors: dict, path: str):
    """Salva os tensores comprimidos no formato .cromv3 customizado para LLMs SOTA."""
    MAGIC = b'CROM'
    VERSION = 3
    
    config_json = json.dumps(config_dict).encode('utf-8')
    
    with open(path, 'wb') as f:
        # HEADER
        f.write(MAGIC)
        f.write(struct.pack('<H', VERSION))
        f.write(struct.pack('<I', len(config_json)))
        f.write(config_json)
        
        # Salvamos o dicionário comprimido inteiro
        # Para cada tensor no dict, escrevemos o nome, shape, dtype e binário
        f.write(struct.pack('<I', len(tensors)))
        
        for name, tensor in tensors.items():
            name_b = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_b)))
            f.write(name_b)
            
            # Metadata: shape len, shape, dtype_code
            f.write(struct.pack('<I', len(tensor.shape)))
            for s in tensor.shape:
                f.write(struct.pack('<I', s))
                
            # Dtypes: 0 = float32, 1 = float16, 2 = uint16, 3 = int64
            if tensor.dtype == torch.float32:
                dt_code = 0
            elif tensor.dtype == torch.float16:
                dt_code = 1
            elif tensor.dtype == torch.uint16 or tensor.dtype == torch.int16:
                dt_code = 2
            else:
                dt_code = 3
            f.write(struct.pack('<I', dt_code))
            
            # Dados
            np_array = tensor.cpu().numpy()
            f.write(np_array.tobytes())


def ptq_compress_model(hf_model_id: str, output_path: str, K: int = 256, D: int = 8):
    """
    Faz o download do modelo HF, comprime todas as camadas nn.Linear,
    e salva o `.cromv3`.
    """
    print(f"🚀 Iniciando Crom PTQ (Post-Training Quantization) para {hf_model_id}")
    print(f"📊 Hyperparams da Compressão: K={K} (Centroides), D={D} (Block Size)")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("⏳ Baixando / Carregando modelo FP16 na RAM/VRAM...")
    t0 = time.time()
    
    # Usa device_map='auto' para evitar OOM se possível, ou carrega pra CPU
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        device_map="auto" if device == "cuda" else None
    )
    t1 = time.time()
    print(f"✅ Modelo carregado em {t1-t0:.1f}s")
    
    compressed_state_dict = {}
    total_original_bytes = 0
    total_compressed_bytes = 0
    
    # Itera sobre todas as camadas do modelo
    state_dict = model.state_dict()
    for name, tensor in state_dict.items():
        original_bytes = tensor.numel() * tensor.element_size()
        total_original_bytes += original_bytes
        
        is_linear_weight = len(tensor.shape) == 2 and "embed" not in name and "norm" not in name
        is_ffn = "up_proj" in name or "down_proj" in name or "gate_proj" in name
        
        if is_linear_weight and is_ffn:
            print(f"  [PTQ Híbrido] Comprimindo FFN {name} {list(tensor.shape)}...")
            
            codebook, indices, mse = compress_linear(tensor, K=K, D=D)
            
            # Armazena os tensores comprimidos com sufixos
            compressed_state_dict[f"{name}.crom_codebook"] = codebook
            compressed_state_dict[f"{name}.crom_indices"] = indices
            
            comp_bytes = codebook.numel() * codebook.element_size() + indices.numel() * 2
            total_compressed_bytes += comp_bytes
            
            print(f"      ↳ MSE: {mse:.4f} | Size: {original_bytes/1e6:.1f}MB → {comp_bytes/1e6:.1f}MB ({(original_bytes/comp_bytes):.1f}x compressão)")
            
        else:
            # Mantém tensores de Attention, lm_head, Embeddings ou Norms originais em fp16
            if is_linear_weight:
                print(f"  [Nativo FP16] Preservando {name} {list(tensor.shape)} (Atenção/Boca)")
            else:
                print(f"  [Pass] Mantendo {name} {list(tensor.shape)} intacto.")
            tensor_fp16 = tensor.half()
            compressed_state_dict[name] = tensor_fp16
            total_compressed_bytes += tensor_fp16.numel() * tensor_fp16.element_size()

    # Salva no formato Crom V3 SOTA
    print(f"\n💾 Salvando pacote comprimido em {output_path}...")
    
    # Puxamos o config real do HF
    config_dict = model.config.to_dict()
    # Adicionamos os hiperparâmetros da compressão
    config_dict["crom_K"] = K
    config_dict["crom_D"] = D
    config_dict["hf_model_id"] = hf_model_id
    
    save_cromv3_sota(hf_model_id, config_dict, compressed_state_dict, output_path)
    
    print("\n" + "="*50)
    print("🎉 COMPRESSÃO PTQ FINALIZADA!")
    print(f"📉 Tamanho Original: {total_original_bytes / 1e9:.2f} GB")
    print(f"📦 Tamanho Comprimido: {total_compressed_bytes / 1e9:.2f} GB")
    print(f"🔥 Compressão Média: {total_original_bytes / total_compressed_bytes:.1f}x")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CromLinear PTQ Compressor")
    parser.add_argument("--model", type=str, default="microsoft/Phi-3-mini-4k-instruct", help="HF Model ID")
    parser.add_argument("--out", type=str, default="phi3_crom.cromv3", help="Output path")
    parser.add_argument("--k", type=int, default=256, help="K (Number of centroids)")
    parser.add_argument("--d", type=int, default=8, help="D (Block dimension)")
    
    args = parser.parse_args()
    
    ptq_compress_model(args.model, args.out, args.k, args.d)
