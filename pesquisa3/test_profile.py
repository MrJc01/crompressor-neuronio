import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'kernels'))
import torch
import time
from local_chat import load_model_for_inference

model, tokenizer = load_model_for_inference("tiny_teste.cromv3")
model.eval()

input_ids = torch.randint(0, 1000, (1, 1))

# Profile
import time
from collections import defaultdict

times = defaultdict(float)

def get_hook(name):
    def hook(module, input, output):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()
        start = module._start_time
        times[module.__class__.__name__] += (end - start)
    return hook

def pre_hook(module, input):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    module._start_time = time.time()

for name, module in model.named_modules():
    if len(list(module.children())) == 0: # only leaf modules
        module.register_forward_pre_hook(pre_hook)
        module.register_forward_hook(get_hook(name))

print("Starting profile run...")
with torch.no_grad():
    model(input_ids)

print("\n--- Profile Results ---")
total_time = 0
for cls_name, t in sorted(times.items(), key=lambda x: x[1], reverse=True):
    print(f"{cls_name}: {t:.4f}s")
    total_time += t
print(f"Total accounted time: {total_time:.4f}s")
