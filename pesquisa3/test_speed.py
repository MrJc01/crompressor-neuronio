import sys, os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
sys.path.append(os.path.join(os.path.dirname(__file__), 'kernels'))
import torch
import time
from local_chat import load_model_for_inference

model, tokenizer = load_model_for_inference("tiny_teste.cromv3")
model.eval()

input_ids = torch.randint(0, 1000, (1, 1))

t0 = time.time()
with torch.no_grad():
    for _ in range(5):
        out = model(input_ids)
t1 = time.time()
print(f"OMP=1 | Time for 5 forward passes: {t1-t0:.2f}s")
