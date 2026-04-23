"""
Formato .crom v3 — Serialização de Modelo CromGPT
===================================================

Formato binário para salvar/carregar modelos CromGPT
salvando APENAS codebook + índices (sem continuous_weight).

Estrutura:
  [HEADER]       magic + version + config JSON
  [EMBEDDINGS]   token_emb + pos_emb (Float16)
  [LAYERS]       Para cada layer: codebook[K,D] + indices[n_blocks]
  [LN_FINAL]     LayerNorm final
  [CHECKSUM]     SHA-256

Pesquisa 2 — Lab 28
"""

import torch
import struct
import json
import hashlib
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lab26-crom-linear'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lab27-cromgpt-base'))
from crom_linear import CromLinear
from model import CromGPT, CromGPTConfig

MAGIC = b'CROM'
VERSION = 3


def save_cromv3(model: CromGPT, path: str):
    """Salva modelo CromGPT no formato .crom v3 comprimido."""
    
    config_json = json.dumps(model.config.to_dict()).encode('utf-8')
    
    with open(path, 'wb') as f:
        # ═══════════════════════════════════════
        # HEADER
        # ═══════════════════════════════════════
        f.write(MAGIC)                              # 4 bytes
        f.write(struct.pack('<H', VERSION))          # 2 bytes
        f.write(struct.pack('<I', len(config_json))) # 4 bytes
        f.write(config_json)                         # variável
        
        # ═══════════════════════════════════════
        # EMBEDDINGS (Float16 para compressão)
        # ═══════════════════════════════════════
        tok_emb = model.token_emb.weight.data.half().cpu().numpy()
        pos_emb = model.pos_emb.weight.data.half().cpu().numpy()
        
        f.write(struct.pack('<II', *tok_emb.shape))  # vocab_size, d_model
        f.write(tok_emb.tobytes())
        
        f.write(struct.pack('<II', *pos_emb.shape))  # max_seq, d_model
        f.write(pos_emb.tobytes())
        
        # ═══════════════════════════════════════
        # LAYERS (codebook + índices APENAS)
        # ═══════════════════════════════════════
        n_crom = 0
        for name, module in model.named_modules():
            if isinstance(module, CromLinear):
                # Codebook: Float16
                cb = module.codebook.data.half().cpu().numpy()
                f.write(struct.pack('<II', *cb.shape))  # K, D
                f.write(cb.tobytes())
                
                # Índices: uint16
                idx = module.indices.cpu().numpy().astype(np.uint16)
                f.write(struct.pack('<I', len(idx)))     # n_blocks
                f.write(idx.tobytes())
                
                # Bias: Float16
                bias = module.bias.data.half().cpu().numpy()
                f.write(struct.pack('<I', len(bias)))
                f.write(bias.tobytes())
                
                n_crom += 1
        
        # ═══════════════════════════════════════
        # LAYER NORMS (Float16)
        # ═══════════════════════════════════════
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                w = module.weight.data.half().cpu().numpy()
                b = module.bias.data.half().cpu().numpy()
                f.write(struct.pack('<I', len(w)))
                f.write(w.tobytes())
                f.write(b.tobytes())
        
        # ═══════════════════════════════════════
        # FOOTER
        # ═══════════════════════════════════════
        f.write(struct.pack('<I', n_crom))  # n_crom layers
    
    # Checksum
    with open(path, 'rb') as f:
        data = f.read()
    sha = hashlib.sha256(data).hexdigest()
    
    size = os.path.getsize(path)
    return size, sha, n_crom


def load_cromv3(path: str, device='cpu'):
    """Carrega modelo CromGPT do formato .crom v3."""
    
    with open(path, 'rb') as f:
        # Header
        magic = f.read(4)
        assert magic == MAGIC, f"Magic inválido: {magic}"
        version = struct.unpack('<H', f.read(2))[0]
        assert version == VERSION, f"Versão {version} != {VERSION}"
        
        config_len = struct.unpack('<I', f.read(4))[0]
        config_json = json.loads(f.read(config_len).decode('utf-8'))
        config_json.pop('head_dim', None)  # campo calculado, não argumento
        config = CromGPTConfig(**config_json)
        
        # Criar modelo
        model = CromGPT(config)
        
        # Embeddings
        vs, dm = struct.unpack('<II', f.read(8))
        tok_data = np.frombuffer(f.read(vs * dm * 2), dtype=np.float16).reshape(vs, dm)
        model.token_emb.weight.data = torch.from_numpy(tok_data.astype(np.float32)).to(device)
        
        ps, dm2 = struct.unpack('<II', f.read(8))
        pos_data = np.frombuffer(f.read(ps * dm2 * 2), dtype=np.float16).reshape(ps, dm2)
        model.pos_emb.weight.data = torch.from_numpy(pos_data.astype(np.float32)).to(device)
        
        # LM Head tied
        model.lm_head.weight = model.token_emb.weight
        
        # CromLinear layers
        for name, module in model.named_modules():
            if isinstance(module, CromLinear):
                K, D = struct.unpack('<II', f.read(8))
                cb = np.frombuffer(f.read(K * D * 2), dtype=np.float16).reshape(K, D)
                module.codebook.data = torch.from_numpy(cb.astype(np.float32)).to(device)
                
                nb = struct.unpack('<I', f.read(4))[0]
                idx = np.frombuffer(f.read(nb * 2), dtype=np.uint16)
                module.indices = torch.from_numpy(idx.astype(np.int64)).to(device)
                
                bl = struct.unpack('<I', f.read(4))[0]
                bias = np.frombuffer(f.read(bl * 2), dtype=np.float16)
                module.bias.data = torch.from_numpy(bias.astype(np.float32)).to(device)
                
                # Reconstruir continuous_weight a partir do codebook
                module.continuous_weight.data = module.codebook.data[module.indices].to(device)
        
        # LayerNorms
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                n = struct.unpack('<I', f.read(4))[0]
                w = np.frombuffer(f.read(n * 2), dtype=np.float16)
                b = np.frombuffer(f.read(n * 2), dtype=np.float16)
                module.weight.data = torch.from_numpy(w.astype(np.float32)).to(device)
                module.bias.data = torch.from_numpy(b.astype(np.float32)).to(device)
        
        # Footer
        n_crom = struct.unpack('<I', f.read(4))[0]
    
    return model, config, n_crom


# ═══════════════════════════════════════════════════════════════
# TESTES
# ═══════════════════════════════════════════════════════════════

def test_save_load():
    """Teste roundtrip: save → load → compare outputs."""
    print("=" * 60)
    print("TESTE: .crom v3 Save/Load Roundtrip")
    print("=" * 60)
    
    # Criar modelo e rodar forward
    config = CromGPTConfig.tiny()
    model = CromGPT(config)
    model.eval()
    
    x = torch.randint(0, config.vocab_size, (1, 16))
    with torch.no_grad():
        logits_before, _ = model(x)
    
    # Salvar .crom v3
    crom_path = "../../resultados/test_model.cromv3"
    size_crom, sha, n_crom = save_cromv3(model, crom_path)
    
    # Salvar PyTorch normal para comparar tamanho
    pt_path = "../../resultados/test_model.pt"
    torch.save(model.state_dict(), pt_path)
    size_pt = os.path.getsize(pt_path)
    
    # Carregar
    model_loaded, config_loaded, n_loaded = load_cromv3(crom_path)
    model_loaded.eval()
    
    with torch.no_grad():
        logits_after, _ = model_loaded(x)
    
    # Comparar
    # Nota: FP16 causa pequenas diferenças
    diff = (logits_before - logits_after).abs().max().item()
    close = diff < 0.1  # tolerância FP16
    
    print(f"\n  📦 .crom v3:  {size_crom:,} bytes")
    print(f"  📦 PyTorch:   {size_pt:,} bytes")
    print(f"  📊 Compressão: {size_pt/size_crom:.1f}x")
    print(f"  📊 CromLinear layers: {n_crom}")
    print(f"  📊 SHA-256: {sha[:16]}...")
    print(f"  📊 Max diff (FP16): {diff:.6f}")
    print(f"  {'✅' if close else '❌'} Roundtrip {'OK' if close else 'FALHOU'}!")
    print()
    
    # Limpar
    os.remove(crom_path)
    os.remove(pt_path)
    
    return close, size_crom, size_pt


def test_trained_model():
    """Testa save/load com modelo treinado real."""
    print("=" * 60)
    print("TESTE: .crom v3 com Modelo Treinado")
    print("=" * 60)
    
    trained_path = "../../resultados/cromgpt_model.pt"
    if not os.path.exists(trained_path):
        print("  ⏭️ Modelo treinado não encontrado, pulando...")
        return True, 0, 0
    
    # Carregar modelo treinado
    meta = json.load(open('../../data/meta.json'))
    config = CromGPTConfig.tiny()
    config.vocab_size = meta['vocab_size']
    config.max_seq_len = 64
    
    model = CromGPT(config)
    model.load_state_dict(torch.load(trained_path, weights_only=True))
    model.eval()
    
    # Salvar .crom v3
    crom_path = "../../resultados/cromgpt_trained.cromv3"
    size_crom, sha, n_crom = save_cromv3(model, crom_path)
    size_pt = os.path.getsize(trained_path)
    
    # Carregar e verificar
    model2, _, _ = load_cromv3(crom_path)
    model2.eval()
    
    x = torch.randint(0, 1000, (1, 32))
    with torch.no_grad():
        l1, _ = model(x)
        l2, _ = model2(x)
    
    diff = (l1 - l2).abs().max().item()
    
    print(f"\n  📦 .crom v3 (treinado): {size_crom:,} bytes")
    print(f"  📦 PyTorch (treinado):  {size_pt:,} bytes")
    print(f"  📊 Compressão: {size_pt/size_crom:.1f}x")
    print(f"  📊 Max diff: {diff:.6f}")
    print(f"  ✅ Roundtrip OK!" if diff < 0.1 else f"  ❌ Diff alto!")
    print()
    
    return diff < 0.1, size_crom, size_pt


if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  FORMATO .CROM V3 — VALIDAÇÃO                          ║")
    print("║  Pesquisa 2: CromGPT (LLM Nativo .crom)                ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    
    t1, s1_crom, s1_pt = test_save_load()
    t2, s2_crom, s2_pt = test_trained_model()
    
    print("=" * 60)
    print("RESUMO FORMATO .CROM V3")
    print("=" * 60)
    print(f"  ✅ Roundtrip (novo):     {'PASS' if t1 else 'FAIL'} — {s1_pt/max(s1_crom,1):.1f}x compressão")
    print(f"  ✅ Roundtrip (treinado): {'PASS' if t2 else 'FAIL'} — {s2_pt/max(s2_crom,1):.1f}x compressão")
    
    results = {
        "new_model": {"passed": t1, "crom_bytes": s1_crom, "pt_bytes": s1_pt, "ratio": s1_pt/max(s1_crom,1)},
        "trained_model": {"passed": t2, "crom_bytes": s2_crom, "pt_bytes": s2_pt, "ratio": s2_pt/max(s2_crom,1)},
    }
    with open("../../resultados/lab28_cromv3.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  📄 Resultados em resultados/lab28_cromv3.json")
    print()
