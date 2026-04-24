import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'kernels'))
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from local_chat import load_model_for_inference

model, tokenizer = load_model_for_inference("tiny_teste.cromv3")
model.eval()

# Let's check for NaNs in the model's non-linear parameters
has_nan = False
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        print(f"NaN found in parameter {name}")
        has_nan = True

for name, buf in model.named_buffers():
    if torch.isnan(buf).any():
        print(f"NaN found in buffer {name}")
        has_nan = True
    elif torch.isinf(buf).any():
        print(f"Inf found in buffer {name}")
        has_nan = True
    elif buf.dtype in (torch.float32, torch.float16) and buf.abs().max() > 1e4:
        print(f"Extremely large values in buffer {name}: max={buf.abs().max()}")

print("Testing a simple forward pass...")
input_text = "oi"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
with torch.no_grad():
    out = model(input_ids)
    print("Logits shape:", out.logits.shape)
    print("Logits contains NaN?", torch.isnan(out.logits).any().item())
    if torch.isnan(out.logits).any().item():
        print(out.logits)
