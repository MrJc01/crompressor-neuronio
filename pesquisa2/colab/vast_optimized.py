"""
╔══════════════════════════════════════════════════════════════╗
║  CromGPT — Vast.ai (OTIMIZADO para custo mínimo)           ║
║  Objetivo: Máximo resultado com ~$1-2 de compute           ║
║                                                              ║
║  INSTRUÇÕES VAST.AI:                                         ║
║  1. Alugar RTX 3090 (~$0.15-0.25/hr) com PyTorch template   ║
║  2. Abrir terminal (SSH ou Jupyter)                          ║
║  3. Rodar: pip install transformers datasets                 ║
║  4. git clone https://github.com/MrJc01/crompressor-neuronio║
║  5. cd crompressor-neuronio/pesquisa2                        ║
║  6. python colab/vast_optimized.py                           ║
║                                                              ║
║  Tempo estimado: ~2-3h numa RTX 3090                         ║
║  Custo estimado: ~$0.50-0.75                                 ║
╚══════════════════════════════════════════════════════════════╝
"""

import subprocess, sys, os
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers', 'datasets'], check=True)

import torch, torch.nn.functional as F, numpy as np, json, time, hashlib

sys.path.insert(0, 'labs/lab26-crom-linear')
sys.path.insert(0, 'labs/lab27-cromgpt-base')
sys.path.insert(0, 'labs/lab28-crom-v3')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"GPU: {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'}")
if DEVICE == 'cuda':
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram:.1f} GB")

# ═══════════════════════════════════════════════════════
# CONFIG — AJUSTAR PARA ECONOMIZAR
# ═══════════════════════════════════════════════════════
MAX_ARTICLES = 50_000    # 50K artigos (~50M tokens) — metade do Colab
MAX_STEPS = 8_000        # 8K steps (suficiente para convergência)
BATCH_SIZE = 16 if (DEVICE=='cuda' and vram > 20) else 8
SEQ_LEN = 256
EPOCHS = 1               # 1 epoch max, mas com MAX_STEPS como limite
LR = 3e-4
LOG_EVERY = 200
EVAL_EVERY = 2000
SAVE_EVERY = 2000

print(f"\n{'='*50}")
print(f"  BATCH_SIZE={BATCH_SIZE} | SEQ_LEN={SEQ_LEN}")
print(f"  MAX_STEPS={MAX_STEPS} | MAX_ARTICLES={MAX_ARTICLES}")
print(f"  Custo estimado: ~$0.50-0.75")
print(f"{'='*50}\n")

# ═══════════════════════════════════════════════════════
# DOWNLOAD DADOS
# ═══════════════════════════════════════════════════════
from datasets import load_dataset
from transformers import AutoTokenizer

DATA_DIR = 'data_vast'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs('checkpoints_vast', exist_ok=True)
os.makedirs('resultados', exist_ok=True)

TOKENIZER_NAME = 'pierreguillou/gpt2-small-portuguese'
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Verificar se dados já existem (re-run sem re-download)
train_path = f'{DATA_DIR}/train.npy'
if os.path.exists(train_path):
    print("📂 Dados já existem, pulando download...")
    tokens_train = np.load(train_path)
    tokens_val = np.load(f'{DATA_DIR}/val.npy')
    meta = json.load(open(f'{DATA_DIR}/meta.json'))
else:
    print(f"📥 Baixando Wikipedia PT ({MAX_ARTICLES:,} artigos)...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train", streaming=True)

    all_tokens = []
    eos_id = tokenizer.eos_token_id
    seen = set(); n_docs = 0

    for i, ex in enumerate(ds):
        if i >= MAX_ARTICLES: break
        text = ex.get("text", "")
        if len(text) < 200: continue
        h = hashlib.md5(text[:500].encode()).hexdigest()
        if h in seen: continue
        seen.add(h)
        lines = [l.strip() for l in text.split("\n")
                 if len(l.strip()) > 10 and not l.startswith("==")
                 and not l.startswith("{{") and not l.startswith("|")]
        clean = " ".join(lines)
        if len(clean) < 200: continue
        toks = tokenizer.encode(clean, add_special_tokens=False)
        all_tokens.extend(toks); all_tokens.append(eos_id)
        n_docs += 1
        if n_docs % 10000 == 0:
            print(f"  ... {n_docs:,} docs, {len(all_tokens):,} tokens")

    all_tokens = np.array(all_tokens, dtype=np.uint16)
    split = int(len(all_tokens) * 0.95)
    tokens_train = all_tokens[:split]
    tokens_val = all_tokens[split:]
    np.save(f'{DATA_DIR}/train.npy', tokens_train)
    np.save(f'{DATA_DIR}/val.npy', tokens_val)
    meta = {"tokenizer": TOKENIZER_NAME, "vocab_size": tokenizer.vocab_size,
            "total_tokens": len(all_tokens), "train_tokens": len(tokens_train),
            "val_tokens": len(tokens_val), "documents": n_docs}
    json.dump(meta, open(f'{DATA_DIR}/meta.json', 'w'), indent=2)
    print(f"\n✅ {n_docs:,} docs, {len(all_tokens):,} tokens")

print(f"Train: {len(tokens_train):,} | Val: {len(tokens_val):,}")

# ═══════════════════════════════════════════════════════
# MODELOS
# ═══════════════════════════════════════════════════════
from crom_linear import CromLinear
from model import CromGPT, CromGPTConfig

vocab = meta['vocab_size']

# CromGPT small
cfg_c = CromGPTConfig.small(); cfg_c.vocab_size = vocab; cfg_c.max_seq_len = SEQ_LEN
model_c = CromGPT(cfg_c).to(DEVICE)

# Baseline
cfg_b = CromGPTConfig.small(); cfg_b.vocab_size = vocab; cfg_b.max_seq_len = SEQ_LEN
cfg_b.use_crom_attention = False; cfg_b.use_crom_ffn = False
model_b = CromGPT(cfg_b).to(DEVICE)

print(f"CromGPT: {sum(p.numel() for p in model_c.parameters()):,} params")
print(f"Baseline: {sum(p.numel() for p in model_b.parameters()):,} params")

# ═══════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════
class TokenDS(torch.utils.data.Dataset):
    def __init__(self, data, seq_len):
        self.data = data.astype(np.int64); self.sl = seq_len
        self.n = max(1, (len(self.data)-1) // seq_len)
    def __len__(self): return self.n
    def __getitem__(self, i):
        s = i * self.sl; chunk = self.data[s:s+self.sl+1]
        if len(chunk) < self.sl+1:
            chunk = np.pad(chunk, (0, self.sl+1-len(chunk)), constant_values=-1)
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)

train_ds = TokenDS(tokens_train, SEQ_LEN)
val_ds = TokenDS(tokens_val, SEQ_LEN)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)
val_ld = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ═══════════════════════════════════════════════════════
# TREINO
# ═══════════════════════════════════════════════════════
def train(model, name, max_steps):
    cb_params = [p for n, p in model.named_parameters() if 'codebook' in n]
    ot_params = [p for n, p in model.named_parameters() if 'codebook' not in n]
    opt = torch.optim.AdamW([
        {'params': ot_params, 'lr': LR},
        {'params': cb_params, 'lr': LR * 3.0},
    ], weight_decay=0.1, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda', enabled=DEVICE=='cuda')
    warmup = min(500, max_steps//10)

    print(f"\n🚀 TREINO: {name} ({max_steps} steps)")
    model.train()
    losses = []; step = 0; t0 = time.time()

    for epoch in range(EPOCHS):
        for x, y in train_ld:
            if step >= max_steps: break
            x, y = x.to(DEVICE), y.to(DEVICE)

            # LR schedule
            if step < warmup: lr = LR * step / max(1, warmup)
            else:
                d = (step - warmup) / max(1, max_steps - warmup)
                lr = LR * max(0.1, 0.5*(1+np.cos(np.pi*d)))
            for g in opt.param_groups: g['lr'] = lr

            with torch.amp.autocast('cuda', enabled=DEVICE=='cuda'):
                _, loss = model(x, y)
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()

            if step % 100 == 0 and hasattr(model, 'update_codebook_indices'):
                model.update_codebook_indices()

            losses.append(loss.item()); step += 1

            if step % LOG_EVERY == 0:
                dt = time.time()-t0
                tps = BATCH_SIZE*SEQ_LEN*step / max(dt,1)
                cb_str = ""
                if hasattr(model, 'get_codebook_stats'):
                    st = model.get_codebook_stats()
                    if st: cb_str = f" | cb {sum(s['utilization'] for s in st)/len(st):.0%}"
                eta = dt/step * (max_steps-step) / 60
                print(f"  [{name}] {step:5d}/{max_steps} | loss {loss.item():.4f} | "
                      f"ppl {min(np.exp(loss.item()),99999):.0f}{cb_str} | "
                      f"{tps:.0f} tok/s | ETA {eta:.0f}min")

            if step % SAVE_EVERY == 0:
                torch.save(model.state_dict(), f'checkpoints_vast/{name}_step{step}.pt')

            if step % EVAL_EVERY == 0 and step > 0:
                model.eval(); vl = []
                with torch.no_grad():
                    for vx, vy in val_ld:
                        vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                        _, vloss = model(vx, vy); vl.append(vloss.item())
                        if len(vl) >= 100: break
                val_avg = np.mean(vl)
                print(f"  [{name}] VAL: {val_avg:.4f} | PPL {np.exp(min(val_avg,20)):.0f}")
                model.train()

        if step >= max_steps: break

    dt = time.time()-t0
    torch.save(model.state_dict(), f'checkpoints_vast/{name}_final.pt')
    print(f"  [{name}] Fim: {dt/60:.1f}min | Loss {losses[0]:.4f}→{losses[-1]:.4f}")
    return losses

# TREINAR AMBOS
losses_c = train(model_c, "CromGPT", MAX_STEPS)
losses_b = train(model_b, "Baseline", MAX_STEPS)

# ═══════════════════════════════════════════════════════
# AVALIAÇÃO COMPLETA
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  AVALIAÇÃO")
print(f"{'='*60}")

model_c.eval(); model_b.eval()

# PPL
def ppl(model):
    t = []; 
    with torch.no_grad():
        for vx, vy in val_ld:
            vx, vy = vx.to(DEVICE), vy.to(DEVICE)
            _, vl = model(vx, vy); t.append(vl.item())
            if len(t) >= 200: break
    return float(np.exp(np.mean(t)))

ppl_c = ppl(model_c); ppl_b = ppl(model_b)
print(f"  PPL:   CromGPT={ppl_c:.0f}  Baseline={ppl_b:.0f}  Ratio={ppl_c/max(ppl_b,1):.1f}x")

# Speed
def speed(model):
    p = torch.randint(0, 1000, (1, 5)).to(DEVICE)
    torch.cuda.synchronize() if DEVICE=='cuda' else None
    t0 = time.time()
    with torch.no_grad(): model.generate(p, max_new_tokens=100, temperature=0.8)
    torch.cuda.synchronize() if DEVICE=='cuda' else None
    return 100/(time.time()-t0)

sp_c = speed(model_c); sp_b = speed(model_b)
print(f"  Speed: CromGPT={sp_c:.0f}  Baseline={sp_b:.0f} tok/s")

# Disco
from crom_v3 import save_cromv3
torch.save(model_b.state_dict(), 'checkpoints_vast/baseline.pt')
sz_base = os.path.getsize('checkpoints_vast/baseline.pt')
sz_crom, sha, nl = save_cromv3(model_c, 'checkpoints_vast/cromgpt.cromv3')
sz_pt = os.path.getsize('checkpoints_vast/CromGPT_final.pt')
print(f"  Disco: .cromv3={sz_crom/1e6:.1f}MB  .pt(base)={sz_base/1e6:.1f}MB  Comp={sz_base/sz_crom:.1f}x")

# 10 Prompts
PROMPTS = ["O Brasil é", "A inteligência artificial", "A cidade de São Paulo",
           "O futebol brasileiro", "A educação no Brasil", "O planeta Terra",
           "A música brasileira", "O Rio de Janeiro", "A tecnologia moderna",
           "A história do Brasil"]

print(f"\n--- GERAÇÃO (10 prompts) ---")
samples = []
for p in PROMPTS:
    ids = tokenizer.encode(p, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        oc = model_c.generate(ids, max_new_tokens=60, temperature=0.7, top_k=40)
        ob = model_b.generate(ids, max_new_tokens=60, temperature=0.7, top_k=40)
    tc = tokenizer.decode(oc[0], skip_special_tokens=True)
    tb = tokenizer.decode(ob[0], skip_special_tokens=True)
    samples.append({'prompt': p, 'cromgpt': tc, 'baseline': tb})
    print(f"\n  [{p}]")
    print(f"  CROM: {tc[:120]}")
    print(f"  BASE: {tb[:120]}")

# Diversidade
def div(texts):
    w = ' '.join(texts).split()
    return len(set(w))/max(len(w),1)
dc = div([s['cromgpt'] for s in samples])
db = div([s['baseline'] for s in samples])

# SALVAR TUDO
results = {
    "dataset": meta, "max_steps": MAX_STEPS,
    "ppl": {"cromgpt": ppl_c, "baseline": ppl_b},
    "speed": {"cromgpt": sp_c, "baseline": sp_b},
    "disk": {"cromv3_bytes": sz_crom, "pt_base_bytes": sz_base, "compression": sz_base/sz_crom},
    "diversity": {"cromgpt": dc, "baseline": db},
    "loss_curve": {"cromgpt_start": losses_c[0], "cromgpt_end": losses_c[-1],
                   "baseline_start": losses_b[0], "baseline_end": losses_b[-1]},
    "samples": samples,
}
with open('resultados/vast_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)

print(f"\n{'='*60}")
print(f"  RESUMO FINAL")
print(f"{'='*60}")
print(f"  PPL:      CromGPT={ppl_c:.0f}  Baseline={ppl_b:.0f}")
print(f"  Speed:    CromGPT={sp_c:.0f}  Baseline={sp_b:.0f} tok/s")
print(f"  Disco:    .cromv3 = {sz_base/sz_crom:.1f}x menor que baseline")
print(f"  Diverse:  CromGPT={dc:.3f}  Baseline={db:.3f}")
print(f"  Loss:     CromGPT {losses_c[0]:.2f}→{losses_c[-1]:.2f}")
print(f"  Loss:     Baseline {losses_b[0]:.2f}→{losses_b[-1]:.2f}")
print(f"\n  ✅ resultados/vast_results.json")
print(f"  ✅ checkpoints_vast/cromgpt.cromv3")
print(f"\n  COPIE ESTES ARQUIVOS ANTES DE DESTRUIR A INSTÂNCIA!")
