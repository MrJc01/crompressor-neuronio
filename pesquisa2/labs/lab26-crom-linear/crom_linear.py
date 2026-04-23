"""
CromLinear — Camada Neural com Pesos-Codebook Nativos
=====================================================

Substitui nn.Linear: em vez de W (Float32 matrix), os pesos são
um codebook C[K, D] + índices I[n_blocks].

Forward:  W = C[I].reshape(in, out);  y = x @ W + b
Backward: Straight-Through Estimator (gradiente ignora quantização)

Pesquisa 2 — CromGPT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from dataclasses import dataclass


class CromLinear(nn.Module):
    """
    Camada Linear onde os pesos são representados por um codebook + índices.
    
    Em vez de armazenar W[in_features, out_features] como Float32,
    dividimos W em blocos de tamanho D e representamos cada bloco
    como um índice apontando para um centróide no codebook.
    
    Compressão: (in*out*4 bytes) → (K*D*4 + n_blocks*2 bytes)
    Para 768x768, K=256, D=64: 2.36MB → 100KB (~23x menor)
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 num_codes: int = 256, code_dim: int = 64,
                 bias: bool = True, commitment_cost: float = 0.25):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_codes = num_codes  # K
        self.code_dim = code_dim    # D
        self.commitment_cost = commitment_cost
        
        # Total de elementos na matriz de pesos
        total_elements = in_features * out_features
        
        # Padding para que total_elements seja divisível por D
        self.padded_elements = math.ceil(total_elements / code_dim) * code_dim
        self.n_blocks = self.padded_elements // code_dim
        
        # ═══════════════════════════════════════════════════
        # PARÂMETROS TREINÁVEIS
        # ═══════════════════════════════════════════════════
        
        # Codebook: K centróides de D dimensões
        # Inicialização: Normal(0, 0.02) — padrão GPT-2
        self.codebook = nn.Parameter(
            torch.randn(num_codes, code_dim) * 0.02
        )
        
        # Pesos contínuos (shadow weights) — usados para calcular gradientes
        # Estes são os pesos "reais" que recebem gradientes via STE
        self.continuous_weight = nn.Parameter(
            torch.randn(self.n_blocks, code_dim) * 0.02
        )
        
        # Índices: qual centróide cada bloco usa (NÃO treinável — calculado)
        self.register_buffer(
            'indices', 
            torch.zeros(self.n_blocks, dtype=torch.long)
        )
        
        # Bias (opcional)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # EMA para atualização do codebook (como VQ-VAE)
        self.register_buffer('ema_count', torch.zeros(num_codes))
        self.register_buffer('ema_weight', torch.zeros(num_codes, code_dim))
        self.ema_decay = 0.99
        
        # Estatísticas
        self._codebook_usage = 0.0
        self._commitment_loss = 0.0
        
        # Inicialização
        self._initialize()
    
    def _initialize(self):
        """Inicializa codebook e atribui índices iniciais."""
        # Inicialização Kaiming para continuous_weight
        nn.init.kaiming_uniform_(self.continuous_weight, a=math.sqrt(5))
        
        # Inicializa codebook com subset dos pesos
        if self.n_blocks >= self.num_codes:
            # Seleciona K blocos aleatórios como centróides iniciais
            perm = torch.randperm(self.n_blocks)[:self.num_codes]
            self.codebook.data.copy_(self.continuous_weight.data[perm])
        else:
            # Menos blocos que centróides — inicializa aleatório
            nn.init.kaiming_uniform_(self.codebook, a=math.sqrt(5))
        
        # Atribui índices iniciais via nearest neighbor
        self._update_indices()
    
    @torch.no_grad()
    def _update_indices(self):
        """Recalcula quais centróides cada bloco usa (nearest neighbor)."""
        # Distâncias: [n_blocks, K]
        dists = torch.cdist(
            self.continuous_weight.data, 
            self.codebook.data
        )
        self.indices.copy_(dists.argmin(dim=1))
    
    @torch.no_grad()
    def _update_codebook_ema(self):
        """Atualiza codebook via Exponential Moving Average (como VQ-VAE)."""
        # Conta quantos blocos usam cada centróide
        encodings = F.one_hot(self.indices, self.num_codes).float()  # [n_blocks, K]
        
        # EMA do count
        self.ema_count.mul_(self.ema_decay).add_(
            encodings.sum(0), alpha=1 - self.ema_decay
        )
        
        # EMA dos vetores
        dw = encodings.t() @ self.continuous_weight.data  # [K, D]
        self.ema_weight.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)
        
        # Atualiza codebook
        n = self.ema_count.unsqueeze(1)
        # Laplace smoothing para evitar divisão por zero
        n = (n + 1e-5) / (n.sum() + self.num_codes * 1e-5) * n.sum()
        self.codebook.data.copy_(self.ema_weight / n)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass com Straight-Through Estimator.
        
        1. Quantiza continuous_weight → codebook[indices]
        2. STE: forward usa quantizado, backward usa contínuo
        3. Reconstrói matriz W e faz y = x @ W + b
        """
        # ═══════════════════════════════════════════════════
        # STRAIGHT-THROUGH ESTIMATOR
        # ═══════════════════════════════════════════════════
        
        # Pesos quantizados: lookup no codebook
        quantized = self.codebook[self.indices]  # [n_blocks, D]
        
        # STE: forward usa quantized, backward flui para continuous_weight
        # O truque: adicionar (quantized - continuous) mas detach o gradiente
        w_ste = self.continuous_weight + (quantized - self.continuous_weight).detach()
        
        # ═══════════════════════════════════════════════════
        # RECONSTRUIR MATRIZ W
        # ═══════════════════════════════════════════════════
        
        # Flatten e cortar padding
        w_flat = w_ste.reshape(-1)[:self.in_features * self.out_features]
        W = w_flat.reshape(self.in_features, self.out_features)
        
        # ═══════════════════════════════════════════════════
        # MULTIPLICAÇÃO (idêntica a nn.Linear)
        # ═══════════════════════════════════════════════════
        
        output = x @ W
        if self.bias is not None:
            output = output + self.bias
        
        # ═══════════════════════════════════════════════════
        # LOSSES AUXILIARES (commitment + codebook)
        # ═══════════════════════════════════════════════════
        
        if self.training:
            # Commitment loss: força continuous_weight a ficar perto do codebook
            commitment = F.mse_loss(self.continuous_weight, quantized.detach())
            
            # Codebook loss: puxa codebook para perto dos continuous_weight
            # (este SIM dá gradiente ao codebook)
            codebook_loss = F.mse_loss(quantized, self.continuous_weight.detach())
            
            self._commitment_loss = self.commitment_cost * commitment + codebook_loss
            
            # Estatísticas de utilização
            unique_codes = len(torch.unique(self.indices))
            self._codebook_usage = unique_codes / self.num_codes
        
        return output
    
    def get_auxiliary_loss(self) -> torch.Tensor:
        """Retorna commitment + codebook loss para adicionar à loss principal."""
        return self._commitment_loss if isinstance(self._commitment_loss, torch.Tensor) else torch.tensor(0.0)
    
    def get_codebook_utilization(self) -> float:
        """Retorna % de centróides sendo usados (0.0 a 1.0)."""
        return self._codebook_usage
    
    def get_compression_ratio(self) -> float:
        """Retorna taxa de compressão vs nn.Linear equivalente."""
        original = self.in_features * self.out_features * 4  # Float32
        compressed = (self.num_codes * self.code_dim * 4 +   # Codebook
                      self.n_blocks * 2)                     # Índices (uint16)
        return original / compressed
    
    def extra_repr(self) -> str:
        return (
            f'in={self.in_features}, out={self.out_features}, '
            f'K={self.num_codes}, D={self.code_dim}, '
            f'blocks={self.n_blocks}, '
            f'compression={self.get_compression_ratio():.1f}x, '
            f'bias={self.bias is not None}'
        )


# ═══════════════════════════════════════════════════════════════
# TESTES DE VALIDAÇÃO
# ═══════════════════════════════════════════════════════════════

def test_basic_forward():
    """Teste 0: Verifica que CromLinear instancia e roda forward/backward."""
    print("=" * 60)
    print("TESTE 0: Forward + Backward básico")
    print("=" * 60)
    
    layer = CromLinear(32, 16, num_codes=16, code_dim=8)
    x = torch.randn(4, 32)  # batch=4
    
    y = layer(x)
    # Loss principal + auxiliary (commitment + codebook loss)
    loss = y.sum() + layer.get_auxiliary_loss()
    loss.backward()
    
    assert y.shape == (4, 16), f"Shape errado: {y.shape}"
    assert layer.continuous_weight.grad is not None, "Continuous weight sem gradiente (STE falhou)!"
    assert layer.codebook.grad is not None, "Codebook sem gradiente (codebook_loss falhou)!"
    
    print(f"  ✅ Shape: {y.shape}")
    print(f"  ✅ Continuous grad norm: {layer.continuous_weight.grad.norm():.6f}")
    print(f"  ✅ Codebook grad norm: {layer.codebook.grad.norm():.6f}")
    print(f"  ✅ Compressão: {layer.get_compression_ratio():.1f}x")
    print(f"  ✅ Codebook utilização: {layer.get_codebook_utilization():.1%}")
    print()
    return True


def test_regression():
    """Teste 1: CromLinear aprende regressão linear y = W·x + b."""
    print("=" * 60)
    print("TESTE 1: Regressão Linear (y = Wx + b)")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Target: uma transformação linear conhecida
    W_true = torch.randn(32, 16) * 0.5
    b_true = torch.randn(16) * 0.1
    
    # Dados
    X = torch.randn(256, 32)
    Y = X @ W_true + b_true
    
    # Modelo CromLinear (LR separado para codebook)
    model_crom = CromLinear(32, 16, num_codes=32, code_dim=8)
    opt_crom = torch.optim.Adam([
        {'params': [model_crom.continuous_weight], 'lr': 3e-3},
        {'params': [model_crom.codebook], 'lr': 1e-2},
        {'params': [model_crom.bias], 'lr': 3e-3},
    ])
    
    # Modelo baseline
    model_base = nn.Linear(32, 16)
    opt_base = torch.optim.Adam(model_base.parameters(), lr=3e-3)
    
    losses_crom = []
    losses_base = []
    
    for step in range(2000):
        # CromLinear
        opt_crom.zero_grad()
        pred_crom = model_crom(X)
        loss_crom = F.mse_loss(pred_crom, Y) + model_crom.get_auxiliary_loss()
        loss_crom.backward()
        opt_crom.step()
        
        # Re-assign índices periodicamente
        if step % 50 == 0:
            model_crom._update_indices()
        
        # Baseline
        opt_base.zero_grad()
        pred_base = model_base(X)
        loss_base = F.mse_loss(pred_base, Y)
        loss_base.backward()
        opt_base.step()
        
        losses_crom.append(F.mse_loss(pred_crom, Y).item())
        losses_base.append(loss_base.item())
    
    final_crom = losses_crom[-1]
    final_base = losses_base[-1]
    
    # Threshold de 0.5 é realista para codebook VQ
    status = "✅" if final_crom < 0.5 else "❌"
    print(f"  {status} CromLinear loss final: {final_crom:.6f}")
    print(f"  ✅ Baseline loss final:  {final_base:.6f}")
    print(f"  📊 Ratio: {final_crom/max(final_base, 1e-10):.1f}x")
    print(f"  📊 Codebook utilização: {model_crom.get_codebook_utilization():.1%}")
    print(f"  📊 Compressão: {model_crom.get_compression_ratio():.1f}x")
    print()
    
    return final_crom < 0.5


def test_xor():
    """Teste 2: MLP com CromLinear resolve XOR."""
    print("=" * 60)
    print("TESTE 2: XOR (não-linear)")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Dados XOR
    X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
    Y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)
    
    # Modelo CromLinear
    model_crom = nn.Sequential(
        CromLinear(2, 32, num_codes=8, code_dim=4),
        nn.ReLU(),
        CromLinear(32, 1, num_codes=8, code_dim=4),
        nn.Sigmoid()
    )
    opt_crom = torch.optim.Adam(model_crom.parameters(), lr=1e-2)
    
    # Baseline
    model_base = nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )
    opt_base = torch.optim.Adam(model_base.parameters(), lr=1e-2)
    
    for step in range(2000):
        # CromLinear
        opt_crom.zero_grad()
        pred_crom = model_crom(X)
        aux_loss = sum(m.get_auxiliary_loss() for m in model_crom if isinstance(m, CromLinear))
        loss_crom = F.binary_cross_entropy(pred_crom, Y) + aux_loss
        loss_crom.backward()
        opt_crom.step()
        
        if step % 200 == 0:
            for m in model_crom:
                if isinstance(m, CromLinear):
                    m._update_indices()
        
        # Baseline
        opt_base.zero_grad()
        pred_base = model_base(X)
        loss_base = F.binary_cross_entropy(pred_base, Y)
        loss_base.backward()
        opt_base.step()
    
    # Avaliar
    with torch.no_grad():
        pred_crom = model_crom(X)
        pred_base = model_base(X)
        acc_crom = ((pred_crom > 0.5).float() == Y).float().mean().item()
        acc_base = ((pred_base > 0.5).float() == Y).float().mean().item()
    
    status = "✅" if acc_crom >= 0.95 else "❌"
    print(f"  {status} CromLinear accuracy: {acc_crom:.0%}")
    print(f"  ✅ Baseline accuracy:  {acc_base:.0%}")
    print(f"  📊 Predições CromLinear: {pred_crom.squeeze().tolist()}")
    print(f"  📊 Predições Baseline:  {pred_base.squeeze().tolist()}")
    print()
    
    return acc_crom >= 0.95


def test_mnist():
    """Teste 3: CromLinear em MNIST (se torchvision disponível)."""
    print("=" * 60)
    print("TESTE 3: MNIST (classificação real)")
    print("=" * 60)
    
    try:
        from torchvision import datasets, transforms
    except ImportError:
        print("  ⚠️ torchvision não disponível — pulando MNIST")
        print("  (instale com: pip install torchvision)")
        print()
        return None
    
    torch.manual_seed(42)
    
    # Dados
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = datasets.MNIST('./.mnist_data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./.mnist_data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=256)
    
    # Modelo CromLinear
    model_crom = nn.Sequential(
        nn.Flatten(),
        CromLinear(784, 256, num_codes=256, code_dim=64),
        nn.ReLU(),
        CromLinear(256, 10, num_codes=64, code_dim=16)
    )
    opt_crom = torch.optim.Adam(model_crom.parameters(), lr=1e-3)
    
    # Baseline
    model_base = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    opt_base = torch.optim.Adam(model_base.parameters(), lr=1e-3)
    
    # Treinar 5 epochs
    for epoch in range(5):
        for batch_x, batch_y in train_loader:
            # CromLinear
            opt_crom.zero_grad()
            logits_crom = model_crom(batch_x)
            aux = sum(m.get_auxiliary_loss() for m in model_crom if isinstance(m, CromLinear))
            loss_crom = F.cross_entropy(logits_crom, batch_y) + aux
            loss_crom.backward()
            opt_crom.step()
            
            # Baseline
            opt_base.zero_grad()
            logits_base = model_base(batch_x)
            loss_base = F.cross_entropy(logits_base, batch_y)
            loss_base.backward()
            opt_base.step()
        
        # Re-assign após cada epoch
        for m in model_crom:
            if isinstance(m, CromLinear):
                m._update_indices()
        
        print(f"  Epoch {epoch+1}/5 — loss_crom: {loss_crom.item():.4f}, loss_base: {loss_base.item():.4f}")
    
    # Avaliar
    correct_crom = 0
    correct_base = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            pred_crom = model_crom(batch_x).argmax(1)
            pred_base = model_base(batch_x).argmax(1)
            correct_crom += (pred_crom == batch_y).sum().item()
            correct_base += (pred_base == batch_y).sum().item()
            total += batch_y.size(0)
    
    acc_crom = correct_crom / total
    acc_base = correct_base / total
    
    # Compressão
    crom_layers = [m for m in model_crom if isinstance(m, CromLinear)]
    compressions = [l.get_compression_ratio() for l in crom_layers]
    
    status = "✅" if acc_crom >= 0.90 else "❌"
    print()
    print(f"  {status} CromLinear accuracy: {acc_crom:.2%}")
    print(f"  ✅ Baseline accuracy:  {acc_base:.2%}")
    print(f"  📊 Perda relativa: {(acc_base - acc_crom)*100:.1f}%")
    print(f"  📊 Compressões: {[f'{c:.1f}x' for c in compressions]}")
    print(f"  📊 Codebook utilização: {[f'{l.get_codebook_utilization():.0%}' for l in crom_layers]}")
    print()
    
    # Salvar resultados
    results = {
        "test": "mnist",
        "crom_accuracy": acc_crom,
        "baseline_accuracy": acc_base,
        "accuracy_gap": acc_base - acc_crom,
        "compressions": compressions,
        "codebook_utilization": [l.get_codebook_utilization() for l in crom_layers],
    }
    
    return acc_crom >= 0.90, results


# ═══════════════════════════════════════════════════════════════
# RUNNER PRINCIPAL
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  CROMLINEAR — VALIDAÇÃO SINTÉTICA                       ║")
    print("║  Pesquisa 2: CromGPT (LLM Nativo .crom)                ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    
    results = {}
    
    # Teste 0: Forward/Backward
    t0 = test_basic_forward()
    results["forward_backward"] = {"passed": t0}
    
    # Teste 1: Regressão
    t1 = test_regression()
    results["regression"] = {"passed": t1}
    
    # Teste 2: XOR
    t2 = test_xor()
    results["xor"] = {"passed": t2}
    
    # Teste 3: MNIST
    mnist_result = test_mnist()
    if mnist_result is not None:
        t3, mnist_data = mnist_result
        results["mnist"] = {"passed": t3, **mnist_data}
    else:
        t3 = None
    
    # Resumo
    print("=" * 60)
    print("RESUMO")
    print("=" * 60)
    passed = sum(1 for v in [t0, t1, t2, t3] if v is True)
    total = sum(1 for v in [t0, t1, t2, t3] if v is not None)
    print(f"  Testes: {passed}/{total} passaram")
    for name, r in results.items():
        status = "✅" if r["passed"] else "❌"
        print(f"  {status} {name}")
    
    # Salvar JSON
    results_path = "../../resultados/lab26_cromlinear.json"
    try:
        import os
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  📄 Resultados salvos em: {results_path}")
    except Exception as e:
        print(f"\n  ⚠️ Erro ao salvar resultados: {e}")
    
    print()
