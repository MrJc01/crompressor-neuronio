"""Baseline comparativo: CromGPT vs nn.Linear puro, lado a lado."""
import numpy as np, torch, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lab26-crom-linear'))
from crom_linear import CromLinear
from model import CromGPT, CromGPTConfig

# Dados
tokens = np.load('../../data/train.npy').astype(np.int64)
val_tokens = np.load('../../data/val.npy').astype(np.int64)
meta = json.load(open('../../data/meta.json'))

# Baseline (nn.Linear puro)
cfg_b = CromGPTConfig.tiny()
cfg_b.vocab_size = meta['vocab_size']; cfg_b.max_seq_len = 64
cfg_b.use_crom_attention = False; cfg_b.use_crom_ffn = False
model_b = CromGPT(cfg_b)

# CromGPT
cfg_c = CromGPTConfig.tiny()
cfg_c.vocab_size = meta['vocab_size']; cfg_c.max_seq_len = 64
model_c = CromGPT(cfg_c)

pb = sum(p.numel() for p in model_b.parameters())
pc = sum(p.numel() for p in model_c.parameters())
print(f'Baseline: {pb:,} params | CromGPT: {pc:,} params')

opt_b = torch.optim.AdamW(model_b.parameters(), lr=5e-4, weight_decay=0.1)
opt_c = torch.optim.AdamW([
    {'params': [p for n, p in model_c.named_parameters() if 'codebook' not in n], 'lr': 5e-4},
    {'params': [p for n, p in model_c.named_parameters() if 'codebook' in n], 'lr': 1.5e-3},
], weight_decay=0.1)

model_b.train(); model_c.train()
lb = []; lc = []; seq = 64
print(f'\nTreinando 500 steps...')

for step in range(500):
    s = np.random.randint(0, len(tokens) - seq - 1)
    ch = tokens[s:s+seq+1]
    x = torch.tensor(ch[:-1]).unsqueeze(0)
    y = torch.tensor(ch[1:]).unsqueeze(0)

    _, lo_b = model_b(x, y); opt_b.zero_grad(); lo_b.backward()
    torch.nn.utils.clip_grad_norm_(model_b.parameters(), 1.0); opt_b.step()

    _, lo_c = model_c(x, y); opt_c.zero_grad(); lo_c.backward()
    torch.nn.utils.clip_grad_norm_(model_c.parameters(), 1.0); opt_c.step()

    if step % 25 == 0: model_c.update_codebook_indices()
    lb.append(lo_b.item()); lc.append(lo_c.item())

    if step % 50 == 0:
        st = model_c.get_codebook_stats()
        u = sum(s2['utilization'] for s2 in st) / len(st)
        print(f'  step {step:3d} | base {lo_b.item():.4f} | crom {lo_c.item():.4f} | gap {lo_c.item()-lo_b.item():+.4f} | cb {u:.0%}')

# Val
model_b.eval(); model_c.eval()
vb2 = []; vc2 = []
with torch.no_grad():
    for i in range(0, min(len(val_tokens)-seq-1, 2000), seq):
        ch = val_tokens[i:i+seq+1]
        if len(ch) < seq+1: break
        x = torch.tensor(ch[:-1]).unsqueeze(0)
        y = torch.tensor(ch[1:]).unsqueeze(0)
        _, v1 = model_b(x, y); vb2.append(v1.item())
        _, v2 = model_c(x, y); vc2.append(v2.item())

vb_m = np.mean(vb2); vc_m = np.mean(vc2)

# Disco
torch.save(model_b.state_dict(), '../../resultados/baseline_model.pt')
torch.save(model_c.state_dict(), '../../resultados/cromgpt_model.pt')
sb = os.path.getsize('../../resultados/baseline_model.pt')
sc = os.path.getsize('../../resultados/cromgpt_model.pt')

print(f'\n{"="*60}')
print(f'  COMPARAÇÃO FINAL (500 steps)')
print(f'{"="*60}')
print(f'  Params:     base={pb:,}  crom={pc:,}')
print(f'  Train loss: base={lb[-1]:.4f}  crom={lc[-1]:.4f}  gap={lc[-1]-lb[-1]:+.4f}')
print(f'  Val loss:   base={vb_m:.4f}  crom={vc_m:.4f}  gap={vc_m-vb_m:+.4f}')
print(f'  Val PPL:    base={np.exp(min(vb_m,20)):.0f}  crom={np.exp(min(vc_m,20)):.0f}')
print(f'  Disco:      base={sb:,}B  crom={sc:,}B  ratio={sb/sc:.2f}x')

results = {
    'baseline': {'params': pb, 'train_loss': lb[-1], 'val_loss': float(vb_m), 'val_ppl': float(np.exp(min(vb_m,20))), 'size_bytes': sb},
    'cromgpt': {'params': pc, 'train_loss': lc[-1], 'val_loss': float(vc_m), 'val_ppl': float(np.exp(min(vc_m,20))), 'size_bytes': sc},
    'gap': {'train_loss': lc[-1]-lb[-1], 'val_loss': float(vc_m-vb_m), 'size_ratio': sb/sc},
    'steps': 500,
}
with open('../../resultados/lab27_baseline_comparison.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f'\n  ✅ Salvos em resultados/lab27_baseline_comparison.json')
