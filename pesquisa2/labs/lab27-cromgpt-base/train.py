"""
Training Loop — CromGPT
========================

Loop de treinamento completo para o CromGPT.
Funciona local (mini dataset) e no Colab (dataset completo).

Pesquisa 2 — Lab 27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
import time
import argparse
from pathlib import Path

# Imports do projeto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lab26-crom-linear'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lab27-cromgpt-base'))
from crom_linear import CromLinear
from model import CromGPT, CromGPTConfig


# ═══════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════

class TokenDataset(torch.utils.data.Dataset):
    """Dataset de tokens pré-tokenizados (numpy array)."""
    
    def __init__(self, data_path: str, seq_len: int = 512):
        self.tokens = np.load(data_path).astype(np.int64)
        self.seq_len = seq_len
        # Número de sequências completas
        self.n_seqs = max(1, (len(self.tokens) - 1) // seq_len)
    
    def __len__(self):
        return self.n_seqs
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 para target
        
        # Clip para não ultrapassar
        end = min(end, len(self.tokens))
        chunk = self.tokens[start:end]
        
        # Pad se necessário
        if len(chunk) < self.seq_len + 1:
            chunk = np.pad(chunk, (0, self.seq_len + 1 - len(chunk)), constant_values=-1)
        
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# ═══════════════════════════════════════════════════════════════
# TRAINER
# ═══════════════════════════════════════════════════════════════

class CromGPTTrainer:
    """Loop de treinamento completo para CromGPT."""
    
    def __init__(self, model, train_dataset, val_dataset, config, args):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.args = args
        self.device = args.device
        
        self.model.to(self.device)
        
        # Optimizer com LR separado para codebook
        param_groups = self._get_param_groups()
        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay
        )
        
        # Scheduler: linear warmup + cosine decay
        self.total_steps = len(train_dataset) // args.batch_size * args.epochs
        self.warmup_steps = min(args.warmup_steps, self.total_steps // 5)
        
        # DataLoaders
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False
        ) if val_dataset else None
        
        # Métricas
        self.history = {
            'train_loss': [], 'val_loss': [], 'lr': [],
            'codebook_util': [], 'tokens_per_sec': [], 'step': []
        }
        self.global_step = 0
    
    def _get_param_groups(self):
        """Separa parâmetros em grupos com LR diferente."""
        codebook_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'codebook' in name:
                codebook_params.append(param)
            else:
                other_params.append(param)
        
        return [
            {'params': other_params, 'lr': self.args.lr},
            {'params': codebook_params, 'lr': self.args.lr * self.args.codebook_lr_mult},
        ]
    
    def _get_lr(self, step):
        """Linear warmup + cosine decay."""
        if step < self.warmup_steps:
            return self.args.lr * step / max(1, self.warmup_steps)
        
        decay_ratio = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
        return self.args.lr * max(0.1, coeff)  # min 10% do LR original
    
    def _update_lr(self, step):
        """Atualiza LR do optimizer."""
        lr = self._get_lr(step)
        for i, group in enumerate(self.optimizer.param_groups):
            if i == 1:  # codebook group
                group['lr'] = lr * self.args.codebook_lr_mult
            else:
                group['lr'] = lr
        return lr
    
    @torch.no_grad()
    def evaluate(self):
        """Calcula loss de validação."""
        if self.val_loader is None:
            return float('nan')
        
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            total_loss += loss.item()
            n_batches += 1
        
        self.model.train()
        return total_loss / max(1, n_batches)
    
    def train(self):
        """Loop de treinamento principal."""
        print(f"\n{'='*60}")
        print(f"  TREINAMENTO CROMGPT")
        print(f"{'='*60}")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {self.args.epochs}")
        print(f"  Batch size: {self.args.batch_size}")
        print(f"  Total steps: {self.total_steps}")
        print(f"  Warmup: {self.warmup_steps}")
        print(f"  LR: {self.args.lr}")
        print(f"  Codebook LR mult: {self.args.codebook_lr_mult}x")
        print(f"{'='*60}\n")
        
        self.model.train()
        best_val_loss = float('inf')
        
        for epoch in range(self.args.epochs):
            epoch_loss = 0
            epoch_tokens = 0
            epoch_start = time.time()
            
            for batch_idx, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                t0 = time.time()
                
                # Forward
                logits, loss = self.model(x, y)
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                
                # Update LR
                lr = self._update_lr(self.global_step)
                
                # Step
                self.optimizer.step()
                
                # Re-assign codebook indices periodicamente
                if self.global_step % self.args.reassign_every == 0:
                    self.model.update_codebook_indices()
                
                # Métricas
                dt = time.time() - t0
                tokens = x.numel()
                tps = tokens / max(dt, 1e-6)
                epoch_loss += loss.item()
                epoch_tokens += tokens
                
                # Log
                if self.global_step % self.args.log_every == 0:
                    stats = self.model.get_codebook_stats()
                    avg_util = sum(s['utilization'] for s in stats) / len(stats) if stats else 0
                    
                    ppl = np.exp(min(loss.item(), 20))  # Cap para evitar overflow
                    
                    self.history['train_loss'].append(loss.item())
                    self.history['lr'].append(lr)
                    self.history['codebook_util'].append(avg_util)
                    self.history['tokens_per_sec'].append(tps)
                    self.history['step'].append(self.global_step)
                    
                    print(
                        f"  step {self.global_step:5d} | "
                        f"loss {loss.item():.4f} | "
                        f"ppl {ppl:.1f} | "
                        f"lr {lr:.2e} | "
                        f"cb {avg_util:.0%} | "
                        f"{tps:.0f} tok/s"
                    )
                
                self.global_step += 1
            
            # Fim da epoch
            epoch_time = time.time() - epoch_start
            avg_epoch_loss = epoch_loss / max(1, batch_idx + 1)
            val_loss = self.evaluate()
            
            self.history['val_loss'].append(val_loss)
            
            print(f"\n  ── Epoch {epoch+1}/{self.args.epochs} ──")
            print(f"     Train loss: {avg_epoch_loss:.4f}")
            print(f"     Val loss:   {val_loss:.4f}")
            print(f"     Tempo:      {epoch_time:.1f}s")
            print(f"     Tokens:     {epoch_tokens:,}\n")
            
            # Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(f"best_model.pt")
            
            self.save_checkpoint(f"checkpoint_epoch{epoch+1}.pt")
        
        # Salvar métricas
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, filename):
        """Salva checkpoint do modelo."""
        os.makedirs(self.args.output_dir, exist_ok=True)
        path = os.path.join(self.args.output_dir, filename)
        torch.save({
            'step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'history': self.history,
        }, path)
    
    def save_history(self):
        """Salva histórico de métricas."""
        os.makedirs(self.args.output_dir, exist_ok=True)
        path = os.path.join(self.args.output_dir, "training_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)


# ═══════════════════════════════════════════════════════════════
# GERAÇÃO DE TEXTO
# ═══════════════════════════════════════════════════════════════

def generate_samples(model, tokenizer, prompts, max_tokens=100, temperature=0.8):
    """Gera texto a partir de prompts usando o modelo treinado."""
    model.eval()
    results = []
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(next(model.parameters()).device)
        output_ids = model.generate(input_ids, max_new_tokens=max_tokens, temperature=temperature)
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        results.append({"prompt": prompt, "output": text})
        print(f"\n  📝 Prompt: {prompt}")
        print(f"  📝 Output: {text[:200]}...")
    
    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento CromGPT")
    
    # Data
    parser.add_argument("--data-dir", default="../../data", help="Diretório de dados")
    parser.add_argument("--output-dir", default="../../checkpoints", help="Diretório de checkpoints")
    
    # Model
    parser.add_argument("--model-size", choices=["tiny", "small"], default="tiny")
    parser.add_argument("--seq-len", type=int, default=64, help="Comprimento da sequência")
    
    # Training
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--codebook-lr-mult", type=float, default=3.0, help="Multiplicador LR codebook")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--reassign-every", type=int, default=50, help="Re-assign codebook a cada N steps")
    parser.add_argument("--log-every", type=int, default=10, help="Log a cada N steps")
    
    # Device
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  CROMGPT — TREINAMENTO                                  ║")
    print("║  Pesquisa 2: CromGPT (LLM Nativo .crom)                ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    
    # Carregar meta
    meta_path = os.path.join(args.data_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    
    print(f"  📊 Dataset: {meta['total_tokens']:,} tokens, vocab {meta['vocab_size']:,}")
    
    # Config
    if args.model_size == "tiny":
        config = CromGPTConfig.tiny()
    else:
        config = CromGPTConfig.small()
    
    config.vocab_size = meta['vocab_size']
    config.max_seq_len = args.seq_len
    
    # Modelo
    model = CromGPT(config)
    params = model.count_parameters()
    print(f"  🧠 Modelo: {args.model_size} — {params['total']:,} params")
    
    # Datasets
    train_ds = TokenDataset(os.path.join(args.data_dir, "train.npy"), seq_len=args.seq_len)
    val_ds = TokenDataset(os.path.join(args.data_dir, "val.npy"), seq_len=args.seq_len)
    print(f"  📁 Train: {len(train_ds)} seqs | Val: {len(val_ds)} seqs")
    
    # Trainer
    trainer = CromGPTTrainer(model, train_ds, val_ds, config, args)
    history = trainer.train()
    
    # Geração de teste
    print(f"\n{'='*60}")
    print(f"  GERAÇÃO DE TEXTO (pós-treino)")
    print(f"{'='*60}")
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(meta['tokenizer'])
    
    prompts = [
        "O Brasil é",
        "A inteligência artificial",
        "A cidade de São Paulo",
    ]
    
    samples = generate_samples(model, tokenizer, prompts, max_tokens=50, temperature=0.8)
    
    # Salvar samples
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "samples.json"), "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    # Resumo final
    print(f"\n{'='*60}")
    print(f"  RESUMO FINAL")
    print(f"{'='*60}")
    print(f"  Loss final (train): {history['train_loss'][-1]:.4f}")
    print(f"  Loss final (val):   {history['val_loss'][-1]:.4f}")
    print(f"  PPL final:          {np.exp(min(history['train_loss'][-1], 20)):.1f}")
    print(f"  Checkpoints em:     {args.output_dir}")
    print()
