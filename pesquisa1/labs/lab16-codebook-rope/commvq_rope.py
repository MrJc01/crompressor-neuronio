import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
import numpy as np

# =============================================================================
# Lab16: CommVQ vs Standard VQ em tensores com RoPE (Rotary Position Embedding)
# =============================================================================

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Calcula as frequências de rotação para cada posição no contexto."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # (end, dim/2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False):
    """
    Aplica RoPE a tensores xq de shape (batch, seq, heads, dim).
    Se inverse=True, desfaz a rotação.
    """
    # Transformar em números complexos (agrupando últimas 2 dimensões)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    
    # Broadcast frequencies: freq_cis tem shape (seq, dim/2)
    # Precisamos (1, seq, 1, dim/2)
    freqs_cis = freqs_cis.view(1, xq_.shape[1], 1, xq_.shape[3])
    
    if inverse:
        # Multiplicar pelo conjugado inverte o ângulo de rotação no plano complexo
        out = xq_ * freqs_cis.conj()
    else:
        out = xq_ * freqs_cis
        
    return torch.view_as_real(out).flatten(3)

def run_experiment():
    print("Iniciando Experimento CommVQ vs VQ Cego...\n")
    
    batch, seq, heads, dim = 1, 1024, 4, 64
    k_clusters = 64
    
    # 1. Gerar 'conceitos semânticos' (5 clusters básicos de significados)
    torch.manual_seed(42)
    semantic_bases = torch.randn(5, dim) * 5
    
    # Preencher a sequência escolhendo um dos 5 conceitos, adicionando ruído
    indices = torch.randint(0, 5, (batch, seq, heads))
    base_tensor = semantic_bases[indices] + torch.randn(batch, seq, heads, dim) * 0.5
    
    # 2. Computar frequências RoPE
    freqs_cis = precompute_freqs_cis(dim, seq)
    
    # 3. O modelo aplica RoPE na geração. Este é o tensor que fica salvo no KV Cache.
    kv_cache_real = apply_rotary_emb(base_tensor, freqs_cis)
    
    print(f"Shape do KV Cache: {kv_cache_real.shape}")
    print(f"Dimensão semântica real: 5 clusters.")
    
    # =========================================================================
    # Cenario A: VQ Cego (Ignora a existência do RoPE, clusteriza matriz bruta)
    # =========================================================================
    flat_cache = kv_cache_real.reshape(-1, dim).numpy()
    
    kmeans_blind = MiniBatchKMeans(n_clusters=k_clusters, random_state=42, n_init=3)
    labels_blind = kmeans_blind.fit_predict(flat_cache)
    reconstructed_blind = torch.tensor(kmeans_blind.cluster_centers_[labels_blind])
    reconstructed_blind = reconstructed_blind.reshape(batch, seq, heads, dim)
    
    error_blind = F.mse_loss(reconstructed_blind, kv_cache_real).item()
    print(f"\n[Cenário A] VQ Cego MSE Error: {error_blind:.4f}")
    
    # =========================================================================
    # Cenario B: CommVQ (Desfaz RoPE -> VQ -> Refaz RoPE)
    # =========================================================================
    # Passo 1: Desrotacionar o KV Cache usando o Inverso
    unrotated_cache = apply_rotary_emb(kv_cache_real, freqs_cis, inverse=True)
    flat_unrotated = unrotated_cache.reshape(-1, dim).numpy()
    
    # Passo 2: Treinar K-Means no espaço SEMÂNTICO (invariante ao tempo)
    kmeans_comm = MiniBatchKMeans(n_clusters=k_clusters, random_state=42, n_init=3)
    labels_comm = kmeans_comm.fit_predict(flat_unrotated)
    reconstructed_unrotated = torch.tensor(kmeans_comm.cluster_centers_[labels_comm])
    reconstructed_unrotated = reconstructed_unrotated.reshape(batch, seq, heads, dim)
    
    # Passo 3: Refazer a rotação na hora da inferência
    reconstructed_commvq = apply_rotary_emb(reconstructed_unrotated, freqs_cis, inverse=False)
    
    error_commvq = F.mse_loss(reconstructed_commvq, kv_cache_real).item()
    print(f"[Cenário B] CommVQ MSE Error:  {error_commvq:.4f}")
    
    # =========================================================================
    # Resultados
    # =========================================================================
    improvement = (error_blind - error_commvq) / error_blind * 100
    print(f"\n✅ Conclusão: O CommVQ reduziu o erro de reconstrução em {improvement:.2f}%")
    print("Motivo: No espaço desrotacionado, o K-Means identificou os 5 conceitos-base originais. No VQ cego, o mesmo conceito na pos 5 e pos 100 pareciam alienígenas para a distância Euclidiana.")

if __name__ == "__main__":
    run_experiment()
