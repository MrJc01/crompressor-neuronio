import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'kernels'))
import torch
from local_chat import load_model_for_inference
from transformers import TextStreamer

model, tokenizer = load_model_for_inference("tiny_teste.cromv3")
model.eval()

input_text = "oi"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
attention_mask = torch.ones_like(input_ids)

print("Starting generation...")
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
try:
    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=10, # small amount to see if it finishes
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer
        )
    print("Generation finished.")
except Exception as e:
    print("Error:", e)
