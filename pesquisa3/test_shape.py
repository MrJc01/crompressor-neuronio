import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'kernels'))
import torch
import time
from local_chat import load_model_for_inference

model, tokenizer = load_model_for_inference("tiny_teste.cromv3")
model.eval()

with torch.no_grad():
    # Warmup
    _ = model(torch.randint(0, 1000, (1, 3)))
    
    t0 = time.time()
    _ = model(torch.randint(0, 1000, (1, 3)))
    t1 = time.time()
    print(f"Time for seq_len=3: {t1-t0:.2f}s")
    
    t0 = time.time()
    _ = model(torch.randint(0, 1000, (1, 1)))
    t1 = time.time()
    print(f"Time for seq_len=1: {t1-t0:.2f}s")
