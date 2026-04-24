import sys
import os
import torch
from transformers import AutoTokenizer

# Adiciona labs ao path para importar as classes nativas
sys.path.append(os.path.join(os.path.dirname(__file__), 'labs/lab27-cromgpt-base'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'labs/lab28-crom-v3'))

from model import CromGPT, CromGPTConfig
from crom_v3 import load_cromv3

def load_model_and_tokenizer():
    print("⏳ Carregando Tokenizer (pierreguillou/gpt2-small-portuguese)...")
    tokenizer = AutoTokenizer.from_pretrained('pierreguillou/gpt2-small-portuguese')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("⏳ Carregando pesos quantizados (.cromv3) - 82 MB apenas!")
    model, _, _ = load_cromv3('resultados/cromgpt_125M.cromv3', device=device)
    model.to(device)
    
    model.eval()
    return model, tokenizer, device

def generate_text(model, tokenizer, device, prompt, max_new_tokens=40):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=50
        )
    
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text

def run_predefined_examples(model, tokenizer, device):
    prompts = [
        "A floresta amazônica",
        "O presidente do Brasil",
        "A economia da América Latina",
        "O futuro do trabalho",
        "Uma descoberta científica",
        "A comida tradicional",
        "As eleições de 2024",
        "O sistema solar",
        "Um buraco negro",
        "A revolução industrial"
    ]
    
    print("\n" + "="*50)
    print("🧠 10 EXEMPLOS DE TESTE - CROMGPT 125M")
    print("="*50)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[Exemplo {i}/10]")
        print(f"🔹 Prompt: {prompt}")
        output = generate_text(model, tokenizer, device, prompt)
        print(f"🔸 CromGPT: {output}")

def interactive_chat(model, tokenizer, device):
    print("\n" + "="*50)
    print("🎙️ CHAT INTERATIVO (Digite 'sair' para encerrar)")
    print("="*50)
    
    while True:
        try:
            prompt = input("\n>> Você: ")
            if prompt.lower() in ['sair', 'exit', 'quit']:
                break
            if not prompt.strip():
                continue
                
            print("Gerando...")
            output = generate_text(model, tokenizer, device, prompt)
            print(f"🤖 CromGPT: {output}")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    model, tokenizer, device = load_model_and_tokenizer()
    
    # 1. Roda os 10 exemplos automáticos
    run_predefined_examples(model, tokenizer, device)
    
    # 2. Entra no modo chat para você testar
    interactive_chat(model, tokenizer, device)
