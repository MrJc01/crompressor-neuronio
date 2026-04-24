import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'kernels'))
import torch
from transformers import AutoTokenizer
from local_chat import load_model_for_inference

model, tokenizer = load_model_for_inference("tiny_teste.cromv3")
model.eval()

# Let's check for NaNs in all parameters and buffers
for name, tensor in model.named_parameters():
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"NaN/Inf found in param: {name}")

for name, tensor in model.named_buffers():
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"NaN/Inf found in buffer: {name}")

print("Checking input processing...")
input_text = "oi"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Register hooks to find NaNs
nan_found = False
def hook_fn(module, input, output, name):
    global nan_found
    if isinstance(output, tuple):
        out = output[0]
    else:
        out = output
    if isinstance(out, torch.Tensor):
        if torch.isnan(out).any():
            print(f"NaN produced by {name} ({module.__class__.__name__})")
            nan_found = True

for name, module in model.named_modules():
    module.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))

print("Running forward pass...")
with torch.no_grad():
    out = model(input_ids)
    print("Logits shape:", out.logits.shape)
    
print("Done.")
