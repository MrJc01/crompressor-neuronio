"""
CromGPT — Transformer Decoder-Only com pesos nativos .crom
==========================================================

Cada nn.Linear do GPT-2 é substituída por CromLinear:
- Attention Q, K, V, O projections → CromLinear
- FFN up_proj, down_proj → CromLinear
- Embeddings e LayerNorm permanecem padrão

Pesquisa 2 — Lab 27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import sys
import os

# Importar CromLinear do lab26
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lab26-crom-linear'))
from crom_linear import CromLinear


# ═══════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════

class CromGPTConfig:
    """Configuração do modelo CromGPT."""
    def __init__(
        self,
        vocab_size: int = 50257,     # GPT-2 tokenizer
        max_seq_len: int = 512,
        n_layers: int = 12,
        n_heads: int = 12,
        d_model: int = 768,
        d_ff: int = 3072,            # 4 * d_model
        dropout: float = 0.1,
        # CromLinear params
        num_codes: int = 256,        # K — centróides no codebook
        code_dim: int = 64,          # D — dimensão de cada centróide
        commitment_cost: float = 0.25,
        # Controle: quais camadas usam CromLinear
        use_crom_attention: bool = True,
        use_crom_ffn: bool = True,
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.use_crom_attention = use_crom_attention
        self.use_crom_ffn = use_crom_ffn
        
        assert d_model % n_heads == 0, "d_model deve ser divisível por n_heads"
        self.head_dim = d_model // n_heads
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def tiny(cls):
        """Config minúscula para testes rápidos."""
        return cls(
            vocab_size=1000, max_seq_len=64, n_layers=2, n_heads=2,
            d_model=64, d_ff=256, num_codes=32, code_dim=16
        )
    
    @classmethod
    def small(cls):
        """Config para treino real (~125M params equivalente)."""
        return cls(
            vocab_size=50257, max_seq_len=512, n_layers=12, n_heads=12,
            d_model=768, d_ff=3072, num_codes=256, code_dim=64
        )


def make_linear(config, in_f, out_f, use_crom=True):
    """Factory: cria CromLinear ou nn.Linear dependendo da config."""
    if use_crom:
        return CromLinear(
            in_f, out_f,
            num_codes=config.num_codes,
            code_dim=config.code_dim,
            commitment_cost=config.commitment_cost
        )
    return nn.Linear(in_f, out_f)


# ═══════════════════════════════════════════════════════════════
# MULTI-HEAD ATTENTION
# ═══════════════════════════════════════════════════════════════

class CromAttention(nn.Module):
    """Multi-Head Attention com CromLinear para Q, K, V, O."""
    
    def __init__(self, config: CromGPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        
        # Q, K, V, O projections — CromLinear ou nn.Linear
        self.q_proj = make_linear(config, config.d_model, config.d_model, config.use_crom_attention)
        self.k_proj = make_linear(config, config.d_model, config.d_model, config.use_crom_attention)
        self.v_proj = make_linear(config, config.d_model, config.d_model, config.use_crom_attention)
        self.o_proj = make_linear(config, config.d_model, config.d_model, config.use_crom_attention)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        # Q, K, V projections
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Causal mask
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Weighted sum
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.o_proj(out)
        out = self.resid_dropout(out)
        
        return out


# ═══════════════════════════════════════════════════════════════
# FEED-FORWARD NETWORK
# ═══════════════════════════════════════════════════════════════

class CromFFN(nn.Module):
    """FFN com CromLinear para up_proj e down_proj."""
    
    def __init__(self, config: CromGPTConfig):
        super().__init__()
        self.up_proj = make_linear(config, config.d_model, config.d_ff, config.use_crom_ffn)
        self.down_proj = make_linear(config, config.d_ff, config.d_model, config.use_crom_ffn)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.up_proj(x)
        x = self.act(x)
        x = self.down_proj(x)
        x = self.dropout(x)
        return x


# ═══════════════════════════════════════════════════════════════
# TRANSFORMER BLOCK
# ═══════════════════════════════════════════════════════════════

class CromBlock(nn.Module):
    """Um bloco Transformer: LayerNorm → Attention → LayerNorm → FFN."""
    
    def __init__(self, config: CromGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CromAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = CromFFN(config)
    
    def forward(self, x, mask=None):
        # Pre-norm (GPT-2 style)
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


# ═══════════════════════════════════════════════════════════════
# MODELO COMPLETO
# ═══════════════════════════════════════════════════════════════

class CromGPT(nn.Module):
    """
    CromGPT: Transformer Decoder-Only com pesos nativos .crom.
    
    Cada nn.Linear é substituída por CromLinear:
    - Forward: pesos são lookups no codebook
    - Backward: Straight-Through Estimator
    """
    
    def __init__(self, config: CromGPTConfig):
        super().__init__()
        self.config = config
        
        # Embeddings (padrão — não CromLinear)
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            CromBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final LayerNorm
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # LM Head (tied com token_emb para economizar params)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.token_emb.weight
        
        # Causal mask (registrada como buffer)
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer('causal_mask', mask.view(1, 1, config.max_seq_len, config.max_seq_len))
        
        # Inicialização
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Inicialização GPT-2 style."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, targets=None):
        """
        Args:
            input_ids: [B, T] token IDs
            targets: [B, T] target IDs (para calcular loss)
        Returns:
            logits: [B, T, vocab_size]
            loss: CrossEntropy + auxiliary losses (se targets fornecido)
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, f"Seq len {T} > max {self.config.max_seq_len}"
        
        # Embeddings
        pos = torch.arange(T, device=input_ids.device)
        tok_emb = self.token_emb(input_ids)      # [B, T, d_model]
        pos_emb = self.pos_emb(pos)                # [T, d_model]
        x = self.emb_dropout(tok_emb + pos_emb)
        
        # Causal mask para esta sequência
        mask = self.causal_mask[:, :, :T, :T]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final norm + LM head
        x = self.ln_f(x)
        logits = self.lm_head(x)                   # [B, T, vocab_size]
        
        # Loss
        loss = None
        if targets is not None:
            # CrossEntropy: next-token prediction
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1
            )
            
            # Auxiliary losses (commitment + codebook)
            aux_loss = self.get_auxiliary_loss()
            loss = loss + aux_loss
        
        return logits, loss
    
    def get_auxiliary_loss(self):
        """Soma todas as auxiliary losses de todas as CromLinear layers."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for module in self.modules():
            if isinstance(module, CromLinear):
                total = total + module.get_auxiliary_loss()
        return total
    
    def update_codebook_indices(self):
        """Re-assign índices em todas as CromLinear layers."""
        for module in self.modules():
            if isinstance(module, CromLinear):
                module._update_indices()
    
    def get_codebook_stats(self):
        """Estatísticas de todas as CromLinear layers."""
        stats = []
        for name, module in self.named_modules():
            if isinstance(module, CromLinear):
                stats.append({
                    'name': name,
                    'utilization': module.get_codebook_utilization(),
                    'compression': module.get_compression_ratio(),
                })
        return stats
    
    def count_parameters(self):
        """Conta parâmetros treináveis vs totais."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        
        # Contar params por tipo
        emb_params = sum(p.numel() for n, p in self.named_parameters() if 'emb' in n)
        crom_params = sum(p.numel() for n, p in self.named_parameters() 
                         if 'codebook' in n or 'continuous' in n)
        ln_params = sum(p.numel() for n, p in self.named_parameters() if 'ln' in n)
        
        return {
            'total': total,
            'trainable': trainable,
            'embedding': emb_params,
            'crom_layers': crom_params,
            'layer_norm': ln_params,
        }

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=50):
        """Gera texto autoregressivamente."""
        self.eval()
        for _ in range(max_new_tokens):
            # Truncar para max_seq_len
            idx_cond = input_ids[:, -self.config.max_seq_len:]
            
            # Forward
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


# ═══════════════════════════════════════════════════════════════
# TESTES DE VALIDAÇÃO
# ═══════════════════════════════════════════════════════════════

def test_cromgpt():
    """Valida que CromGPT instancia e roda forward pass."""
    print("=" * 60)
    print("TESTE: CromGPT Forward Pass")
    print("=" * 60)
    
    # Config tiny para teste rápido
    config = CromGPTConfig.tiny()
    model = CromGPT(config)
    
    # Contagem de parâmetros
    params = model.count_parameters()
    print(f"\n  Parâmetros:")
    for k, v in params.items():
        print(f"    {k}: {v:,}")
    
    # Forward pass
    input_ids = torch.randint(0, config.vocab_size, (2, 32))  # batch=2, seq=32
    targets = torch.randint(0, config.vocab_size, (2, 32))
    
    logits, loss = model(input_ids, targets)
    
    assert logits.shape == (2, 32, config.vocab_size), f"Shape errado: {logits.shape}"
    assert loss is not None, "Loss é None!"
    assert not torch.isnan(loss), "Loss é NaN!"
    
    print(f"\n  ✅ Logits shape: {logits.shape}")
    print(f"  ✅ Loss: {loss.item():.4f}")
    print(f"  ✅ Sem NaN!")
    
    # Backward pass
    loss.backward()
    grad_norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms[name] = p.grad.norm().item()
    
    crom_grads = {k: v for k, v in grad_norms.items() if 'codebook' in k or 'continuous' in k}
    print(f"\n  ✅ Backward ok — {len(grad_norms)} params com gradiente")
    print(f"  ✅ CromLinear grads: {len(crom_grads)} params")
    
    # Codebook stats
    stats = model.get_codebook_stats()
    avg_util = sum(s['utilization'] for s in stats) / len(stats) if stats else 0
    avg_comp = sum(s['compression'] for s in stats) / len(stats) if stats else 0
    print(f"\n  📊 CromLinear layers: {len(stats)}")
    print(f"  📊 Avg codebook utilização: {avg_util:.1%}")
    print(f"  📊 Avg compressão: {avg_comp:.1f}x")
    
    # Geração
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"\n  ✅ Geração: {prompt.shape} → {generated.shape}")
    
    print(f"\n  ✅ TODOS OS TESTES PASSARAM!")
    print()
    
    return True, params, stats


def test_hybrid_mode():
    """Testa modo híbrido: CromLinear só no FFN, Attention com nn.Linear."""
    print("=" * 60)
    print("TESTE: Modo Híbrido (FFN=Crom, Attention=Linear)")
    print("=" * 60)
    
    config = CromGPTConfig.tiny()
    config.use_crom_attention = False  # Attention com nn.Linear
    config.use_crom_ffn = True         # FFN com CromLinear
    
    model = CromGPT(config)
    params = model.count_parameters()
    
    input_ids = torch.randint(0, config.vocab_size, (2, 32))
    targets = torch.randint(0, config.vocab_size, (2, 32))
    
    logits, loss = model(input_ids, targets)
    loss.backward()
    
    stats = model.get_codebook_stats()
    
    print(f"  ✅ Hybrid mode funciona!")
    print(f"  📊 CromLinear layers: {len(stats)} (apenas FFN)")
    print(f"  📊 Total params: {params['total']:,}")
    print()
    
    return True


if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  CROMGPT — VALIDAÇÃO DE ARQUITETURA                    ║")
    print("║  Pesquisa 2: CromGPT (LLM Nativo .crom)                ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    
    # Teste 1: Full CromGPT
    t1, params, stats = test_cromgpt()
    
    # Teste 2: Hybrid mode
    t2 = test_hybrid_mode()
    
    # Resumo
    print("=" * 60)
    print("RESUMO CROMGPT")
    print("=" * 60)
    print(f"  ✅ Full CromGPT: {'PASS' if t1 else 'FAIL'}")
    print(f"  ✅ Hybrid mode: {'PASS' if t2 else 'FAIL'}")
    print(f"  📊 CromLinear layers (full): {len(stats)}")
    print(f"  📊 Total params (tiny): {params['total']:,}")
    print()
    
    # Salvar
    results = {
        "full_cromgpt": {"passed": t1, "params": params},
        "hybrid_mode": {"passed": t2},
        "crom_layers": len(stats),
    }
    
    results_path = "../../resultados/lab27_cromgpt.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  📄 Resultados salvos em: {results_path}")
    print()
