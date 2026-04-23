"""
╔══════════════════════════════════════════════════════════════╗
║  CromGPT — Notebook Colab Completo                          ║
║  Pesquisa 2: Treinamento com GPU T4                         ║
║                                                              ║
║  INSTRUÇÕES:                                                 ║
║  1. Abrir no Google Colab                                    ║
║  2. Runtime → Change runtime type → T4 GPU                   ║
║  3. Executar células em ordem                                ║
╚══════════════════════════════════════════════════════════════╝

Este script cobre TODOS os itens pendentes do PLANEJAMENTO:
- 3.3: Treinamento efetivo com Wikipedia PT
- 3.2.8: Mixed precision (FP16)
- 4.2: Métricas quantitativas (PPL, velocidade, VRAM, disco)
- 4.3: Testes de qualidade (10 prompts PT)
- Baseline comparativo lado a lado
"""

# ═══════════════════════════════════════════════════════════════
# CÉLULA 1: SETUP
# ═══════════════════════════════════════════════════════════════

import subprocess, sys, os

# Clonar repo
if not os.path.exists('crompressor-neuronio'):
    subprocess.run(['git', 'clone', 'https://github.com/MrJc01/crompressor-neuronio.git'], check=True)

os.chdir('crompressor-neuronio/pesquisa2')
sys.path.insert(0, 'labs/lab26-crom-linear')
sys.path.insert(0, 'labs/lab27-cromgpt-base')
sys.path.insert(0, 'labs/lab28-crom-v3')

# Instalar deps
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers', 'datasets'], check=True)

import torch
import torch.nn.functional as F
import numpy as np
import json
import time
import hashlib
from pathlib import Path

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# ═══════════════════════════════════════════════════════════════
# CÉLULA 2: DOWNLOAD WIKIPEDIA PT
# ═══════════════════════════════════════════════════════════════

from datasets import load_dataset
from transformers import AutoTokenizer

DATA_DIR = 'data_colab'
os.makedirs(DATA_DIR, exist_ok=True)

TOKENIZER_NAME = 'pierreguillou/gpt2-small-portuguese'
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

MAX_ARTICLES = 100_000  # ~50-100M tokens
SEQ_LEN = 256

print(f"📥 Baixando Wikipedia PT ({MAX_ARTICLES:,} artigos)...")
ds = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train", streaming=True, trust_remote_code=True)

all_tokens = []
eos_id = tokenizer.eos_token_id
seen = set()
n_docs = 0

for i, ex in enumerate(ds):
    if i >= MAX_ARTICLES:
        break
    text = ex.get("text", "")
    if len(text) < 200:
        continue
    h = hashlib.md5(text[:500].encode()).hexdigest()
    if h in seen:
        continue
    seen.add(h)

    # Limpeza básica
    lines = [l.strip() for l in text.split("\n") 
             if len(l.strip()) > 10 and not l.startswith("==") 
             and not l.startswith("{{") and not l.startswith("|")]
    clean = " ".join(lines)
    if len(clean) < 200:
        continue

    toks = tokenizer.encode(clean, add_special_tokens=False)
    all_tokens.extend(toks)
    all_tokens.append(eos_id)
    n_docs += 1
    
    if n_docs % 10000 == 0:
        print(f"  ... {n_docs:,} docs, {len(all_tokens):,} tokens")

all_tokens = np.array(all_tokens, dtype=np.uint16)
split = int(len(all_tokens) * 0.95)
train_tok = all_tokens[:split]
val_tok = all_tokens[split:]

np.save(f'{DATA_DIR}/train.npy', train_tok)
np.save(f'{DATA_DIR}/val.npy', val_tok)

meta = {
    "tokenizer": TOKENIZER_NAME, "vocab_size": tokenizer.vocab_size,
    "eos_token_id": eos_id, "total_tokens": len(all_tokens),
    "train_tokens": len(train_tok), "val_tokens": len(val_tok),
    "documents": n_docs, "max_seq_len": SEQ_LEN,
}
json.dump(meta, open(f'{DATA_DIR}/meta.json', 'w'), indent=2)

print(f"\n✅ {n_docs:,} docs, {len(all_tokens):,} tokens total")
print(f"✅ Train: {len(train_tok):,} | Val: {len(val_tok):,}")

# ═══════════════════════════════════════════════════════════════
# CÉLULA 3: IMPORTAR MODELOS
# ═══════════════════════════════════════════════════════════════

from crom_linear import CromLinear
from model import CromGPT, CromGPTConfig

meta = json.load(open(f'{DATA_DIR}/meta.json'))

# Config SMALL para treino real
config_crom = CromGPTConfig.small()
config_crom.vocab_size = meta['vocab_size']
config_crom.max_seq_len = SEQ_LEN

# Baseline (nn.Linear puro)
config_base = CromGPTConfig.small()
config_base.vocab_size = meta['vocab_size']
config_base.max_seq_len = SEQ_LEN
config_base.use_crom_attention = False
config_base.use_crom_ffn = False

model_crom = CromGPT(config_crom).to(DEVICE)
model_base = CromGPT(config_base).to(DEVICE)

pc = sum(p.numel() for p in model_crom.parameters())
pb = sum(p.numel() for p in model_base.parameters())
print(f"CromGPT: {pc:,} params | Baseline: {pb:,} params")

# VRAM check
if DEVICE == 'cuda':
    print(f"VRAM usada: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ═══════════════════════════════════════════════════════════════
# CÉLULA 4: DATASET + DATALOADER
# ═══════════════════════════════════════════════════════════════

class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, path, seq_len):
        self.tokens = np.load(path).astype(np.int64)
        self.seq_len = seq_len
        self.n_seqs = max(1, (len(self.tokens) - 1) // seq_len)
    def __len__(self):
        return self.n_seqs
    def __getitem__(self, idx):
        s = idx * self.seq_len
        e = min(s + self.seq_len + 1, len(self.tokens))
        chunk = self.tokens[s:e]
        if len(chunk) < self.seq_len + 1:
            chunk = np.pad(chunk, (0, self.seq_len + 1 - len(chunk)), constant_values=-1)
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)

train_ds = TokenDataset(f'{DATA_DIR}/train.npy', SEQ_LEN)
val_ds = TokenDataset(f'{DATA_DIR}/val.npy', SEQ_LEN)

BATCH_SIZE = 8  # Ajustar se OOM
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {len(train_ds):,} seqs ({len(train_ds)*SEQ_LEN:,} tokens)")
print(f"Val: {len(val_ds):,} seqs")
print(f"Batches/epoch: {len(train_loader):,}")

# ═══════════════════════════════════════════════════════════════
# CÉLULA 5: TRAINING LOOP (COM FP16)
# ═══════════════════════════════════════════════════════════════

def train_model(model, name, train_loader, val_loader, epochs=3, lr=3e-4, 
                codebook_lr_mult=3.0, use_fp16=True):
    """Treina modelo com mixed precision, logging, checkpointing."""
    
    # Optimizer com LR separado para codebook
    codebook_params = [p for n, p in model.named_parameters() if 'codebook' in n]
    other_params = [p for n, p in model.named_parameters() if 'codebook' not in n]
    
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': lr},
        {'params': codebook_params, 'lr': lr * codebook_lr_mult},
    ], weight_decay=0.1, betas=(0.9, 0.95))
    
    scaler = torch.amp.GradScaler('cuda', enabled=(use_fp16 and DEVICE=='cuda'))
    total_steps = len(train_loader) * epochs
    warmup = min(500, total_steps // 10)
    
    history = {'train_loss': [], 'val_loss': [], 'step': [], 'tokens_per_sec': []}
    global_step = 0
    best_val = float('inf')
    
    print(f"\n{'='*60}")
    print(f"  TREINO: {name}")
    print(f"  Epochs: {epochs} | Batch: {BATCH_SIZE} | LR: {lr}")
    print(f"  FP16: {use_fp16} | Steps/epoch: {len(train_loader)}")
    print(f"  Total steps: {total_steps} | Warmup: {warmup}")
    print(f"{'='*60}\n")
    
    model.train()
    for epoch in range(epochs):
        t0 = time.time()
        epoch_loss = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # LR schedule
            if global_step < warmup:
                cur_lr = lr * global_step / max(1, warmup)
            else:
                decay = (global_step - warmup) / max(1, total_steps - warmup)
                cur_lr = lr * max(0.1, 0.5 * (1.0 + np.cos(np.pi * decay)))
            for g in optimizer.param_groups:
                g['lr'] = cur_lr if 'codebook' not in str(g) else cur_lr * codebook_lr_mult
            
            # Forward (FP16)
            with torch.amp.autocast('cuda', enabled=(use_fp16 and DEVICE=='cuda')):
                _, loss = model(x, y)
            
            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Re-assign codebook
            if global_step % 100 == 0 and hasattr(model, 'update_codebook_indices'):
                model.update_codebook_indices()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Log
            if global_step % 200 == 0:
                tps = BATCH_SIZE * SEQ_LEN * 200 / max(time.time() - t0, 1)
                history['train_loss'].append(loss.item())
                history['step'].append(global_step)
                history['tokens_per_sec'].append(tps)
                
                stats_str = ""
                if hasattr(model, 'get_codebook_stats'):
                    stats = model.get_codebook_stats()
                    if stats:
                        util = sum(s['utilization'] for s in stats) / len(stats)
                        stats_str = f" | cb {util:.0%}"
                
                print(f"  [{name}] step {global_step:5d} | loss {loss.item():.4f} | "
                      f"ppl {min(np.exp(loss.item()),99999):.0f} | lr {cur_lr:.2e}{stats_str}")
        
        # Val
        model.eval()
        val_loss = 0; vn = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.amp.autocast('cuda', enabled=(use_fp16 and DEVICE=='cuda')):
                    _, vl = model(x, y)
                val_loss += vl.item(); vn += 1
        val_loss /= max(vn, 1)
        history['val_loss'].append(val_loss)
        model.train()
        
        dt = time.time() - t0
        avg_loss = epoch_loss / max(batch_idx+1, 1)
        print(f"\n  ── {name} Epoch {epoch+1}/{epochs} ──")
        print(f"     Train: {avg_loss:.4f} | Val: {val_loss:.4f} | PPL: {np.exp(min(val_loss,20)):.0f} | {dt:.0f}s")
        
        # Checkpoint
        os.makedirs('checkpoints_colab', exist_ok=True)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f'checkpoints_colab/{name}_best.pt')
            print(f"     💾 Melhor modelo salvo!")
        print()
    
    return history

# ═══ TREINAR AMBOS ═══
print("🚀 Iniciando treinamento...")

# CromGPT
hist_crom = train_model(model_crom, "CromGPT", train_loader, val_loader, 
                         epochs=3, lr=3e-4, use_fp16=True)

# Baseline
hist_base = train_model(model_base, "Baseline", train_loader, val_loader,
                         epochs=3, lr=3e-4, codebook_lr_mult=1.0, use_fp16=True)

# ═══════════════════════════════════════════════════════════════
# CÉLULA 6: AVALIAÇÃO COMPLETA
# ═══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"  AVALIAÇÃO COMPLETA")
print(f"{'='*60}")

# Carregar melhores modelos
model_crom.load_state_dict(torch.load('checkpoints_colab/CromGPT_best.pt', weights_only=True))
model_base.load_state_dict(torch.load('checkpoints_colab/Baseline_best.pt', weights_only=True))
model_crom.eval(); model_base.eval()

# --- PPL ---
def calc_ppl(model, loader):
    total = 0; n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, loss = model(x, y)
            total += loss.item(); n += 1
    return np.exp(total / max(n, 1))

ppl_crom = calc_ppl(model_crom, val_loader)
ppl_base = calc_ppl(model_base, val_loader)
print(f"\n  📊 PPL:  CromGPT={ppl_crom:.0f}  Baseline={ppl_base:.0f}  Ratio={ppl_crom/ppl_base:.1f}x")

# --- Velocidade de Inferência ---
def measure_speed(model, n_tokens=200):
    model.eval()
    prompt = torch.randint(0, 1000, (1, 5)).to(DEVICE)
    torch.cuda.synchronize() if DEVICE=='cuda' else None
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=n_tokens, temperature=0.8)
    torch.cuda.synchronize() if DEVICE=='cuda' else None
    dt = time.time() - t0
    return n_tokens / dt

tps_crom = measure_speed(model_crom)
tps_base = measure_speed(model_base)
print(f"  📊 Speed: CromGPT={tps_crom:.0f} tok/s  Baseline={tps_base:.0f} tok/s")

# --- VRAM ---
if DEVICE == 'cuda':
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        x = torch.randint(0, 1000, (1, SEQ_LEN)).to(DEVICE)
        model_crom(x)
    vram_crom = torch.cuda.max_memory_allocated() / 1e6
    
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model_base(x)
    vram_base = torch.cuda.max_memory_allocated() / 1e6
    print(f"  📊 VRAM: CromGPT={vram_crom:.0f}MB  Baseline={vram_base:.0f}MB")
else:
    vram_crom = vram_base = 0

# --- Disco (.crom v3 vs .pt) ---
from crom_v3 import save_cromv3
torch.save(model_base.state_dict(), 'checkpoints_colab/baseline.pt')
size_base = os.path.getsize('checkpoints_colab/baseline.pt')

crom_path = 'checkpoints_colab/cromgpt.cromv3'
size_crom, sha, n_layers = save_cromv3(model_crom, crom_path)
size_pt = os.path.getsize('checkpoints_colab/CromGPT_best.pt')

print(f"  📊 Disco: .crom v3={size_crom/1e6:.1f}MB  .pt(crom)={size_pt/1e6:.1f}MB  .pt(base)={size_base/1e6:.1f}MB")
print(f"  📊 Compressão .crom vs .pt(base): {size_base/size_crom:.1f}x")

# --- Geração de Texto (10 prompts) ---
PROMPTS = [
    "O Brasil é",
    "A inteligência artificial",
    "A cidade de São Paulo",
    "O futebol brasileiro",
    "A educação no Brasil",
    "O planeta Terra",
    "A música brasileira",
    "O Rio de Janeiro",
    "A tecnologia moderna",
    "A história do Brasil",
]

print(f"\n{'='*60}")
print(f"  GERAÇÃO DE TEXTO (10 prompts)")
print(f"{'='*60}")

samples = []
for prompt in PROMPTS:
    ids = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out_c = model_crom.generate(ids, max_new_tokens=60, temperature=0.7, top_k=40)
        out_b = model_base.generate(ids, max_new_tokens=60, temperature=0.7, top_k=40)
    
    text_c = tokenizer.decode(out_c[0], skip_special_tokens=True)
    text_b = tokenizer.decode(out_b[0], skip_special_tokens=True)
    
    samples.append({'prompt': prompt, 'cromgpt': text_c, 'baseline': text_b})
    print(f"\n  Prompt: {prompt}")
    print(f"  CromGPT:  {text_c[:120]}...")
    print(f"  Baseline: {text_b[:120]}...")

# --- Diversidade Lexical ---
def lexical_diversity(texts):
    all_words = ' '.join(texts).split()
    return len(set(all_words)) / max(len(all_words), 1)

div_crom = lexical_diversity([s['cromgpt'] for s in samples])
div_base = lexical_diversity([s['baseline'] for s in samples])
print(f"\n  📊 Diversidade lexical: CromGPT={div_crom:.3f}  Baseline={div_base:.3f}")

# ═══════════════════════════════════════════════════════════════
# CÉLULA 7: SALVAR RESULTADOS
# ═══════════════════════════════════════════════════════════════

results = {
    "dataset": meta,
    "perplexity": {"cromgpt": ppl_crom, "baseline": ppl_base, "ratio": ppl_crom/ppl_base},
    "speed_tok_s": {"cromgpt": tps_crom, "baseline": tps_base},
    "vram_mb": {"cromgpt": vram_crom, "baseline": vram_base},
    "disk_bytes": {
        "cromv3": size_crom, "pt_crom": size_pt, "pt_base": size_base,
        "compression_vs_base": size_base/size_crom
    },
    "lexical_diversity": {"cromgpt": div_crom, "baseline": div_base},
    "samples": samples,
    "training": {
        "cromgpt": {"final_train_loss": hist_crom['train_loss'][-1] if hist_crom['train_loss'] else None,
                     "final_val_loss": hist_crom['val_loss'][-1] if hist_crom['val_loss'] else None},
        "baseline": {"final_train_loss": hist_base['train_loss'][-1] if hist_base['train_loss'] else None,
                      "final_val_loss": hist_base['val_loss'][-1] if hist_base['val_loss'] else None},
    },
    "config_crom": config_crom.to_dict(),
    "config_base": config_base.to_dict(),
}

os.makedirs('resultados', exist_ok=True)
with open('resultados/colab_full_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)

print(f"\n{'='*60}")
print(f"  RESUMO FINAL")
print(f"{'='*60}")
print(f"  Dataset:     {meta['total_tokens']:,} tokens")
print(f"  PPL:         CromGPT={ppl_crom:.0f}  Baseline={ppl_base:.0f}")
print(f"  Speed:       CromGPT={tps_crom:.0f}  Baseline={tps_base:.0f} tok/s")
print(f"  Disco:       .crom v3 = {size_base/size_crom:.1f}x menor que baseline .pt")
print(f"  Diversidade: CromGPT={div_crom:.3f}  Baseline={div_base:.3f}")
print(f"\n  ✅ Resultados em: resultados/colab_full_results.json")
print(f"  ✅ Modelo .crom v3 em: {crom_path}")
print()

# Download para local
try:
    from google.colab import files
    files.download('resultados/colab_full_results.json')
    files.download(crom_path)
    print("  📥 Arquivos baixados automaticamente!")
except:
    print("  ℹ️ Não está no Colab — resultados salvos localmente.")
