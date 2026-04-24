import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import struct
import json
import numpy as np
import time
import os
import sys

# O Caminho B abandona o kernel C++ customizado.
# Faremos toda a reconstrução da quantização usando o backend nativo do PyTorch (OneBLAS/MKL),
# que é assustadoramente rápido e não causa deadlocks ou falhas de compilação.


class CromLinearSOTA(nn.Module):
    """
    Camada wrapper que substitui o nn.Linear do HuggingFace.
    (Caminho B) Usa reconstrução On-The-Fly puramente em PyTorch para aproveitar o backend OneBLAS.
    """
    def __init__(self, in_features, out_features, K, D, codebook, indices):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        self.D = D
        
        self.register_buffer('codebook', codebook.to(torch.float16).contiguous())
        # O PyTorch exige índices int64 para slicing
        self.register_buffer('indices', indices.to(torch.int64).contiguous())
        
    def forward(self, x):
        orig_dtype = x.dtype
        x_fp16 = x.to(torch.float16)
        
        # A MÁGICA (Caminho B): Reconstrução on-the-fly!
        # indices shape: [out_features, in_features // D]
        # codebook shape: [K, D]
        # W_q shape inicial: [out_features, in_features // D, D]
        W_q = self.codebook[self.indices]
        W_q = W_q.view(self.out_features, self.in_features)
        
        # Delega o cálculo pesado para o F.linear (que invoca o backend nativo do PyTorch / AVX512)
        import torch.nn.functional as F
        out = F.linear(x_fp16, W_q)
        
        return out.to(orig_dtype)


def load_cromv3_sota_file(path: str):
    """Lê o binário customizado que exportamos na Vast.ai e extrai os tensores."""
    MAGIC = b'CROM'
    VERSION = 3
    
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == MAGIC, "Arquivo inválido"
        version = struct.unpack('<H', f.read(2))[0]
        
        config_len = struct.unpack('<I', f.read(4))[0]
        config_json = json.loads(f.read(config_len).decode('utf-8'))
        
        num_tensors = struct.unpack('<I', f.read(4))[0]
        
        tensors = {}
        for _ in range(num_tensors):
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')
            
            shape_len = struct.unpack('<I', f.read(4))[0]
            shape = []
            for _ in range(shape_len):
                shape.append(struct.unpack('<I', f.read(4))[0])
                
            dt_code = struct.unpack('<I', f.read(4))[0]
            
            # Lê os bytes baseados no shape e dtype
            numel = np.prod(shape)
            if dt_code == 0:
                dtype = np.float32; torch_dtype = torch.float32; bytes_per = 4
            elif dt_code == 1:
                dtype = np.float16; torch_dtype = torch.float16; bytes_per = 2
            elif dt_code == 2:
                dtype = np.uint16; torch_dtype = torch.int16; bytes_per = 2
            else:
                dtype = np.int64; torch_dtype = torch.int64; bytes_per = 8
                
            raw_data = f.read(int(numel * bytes_per))
            np_arr = np.frombuffer(raw_data, dtype=dtype).copy().reshape(shape)
            
            # Hack de conversão uint16 -> int16 pro pytorch
            if dt_code == 2:
                np_arr = np_arr.view(np.int16)
                
            tensors[name] = torch.from_numpy(np_arr).to(torch_dtype)
            
    return config_json, tensors


def inject_cromlinear_layers(model, compressed_state_dict, K, D):
    """Função recursiva para trocar os nn.Linear pelos nossos CromLinearSOTA."""
    replaced_count = 0
    
    def replace_recursive(module, prefix=""):
        nonlocal replaced_count
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if f"{full_name}.weight.crom_codebook" in compressed_state_dict:
                codebook = compressed_state_dict[f"{full_name}.weight.crom_codebook"]
                indices = compressed_state_dict[f"{full_name}.weight.crom_indices"]
                
                # Ignoramos camadas Lineares não suportadas pelo PTQ por enquanto
                if not isinstance(child, nn.Linear):
                    replace_recursive(child, full_name)
                    continue
                    
                new_layer = CromLinearSOTA(child.in_features, child.out_features, K, D, codebook, indices)
                setattr(module, name, new_layer)
                replaced_count += 1
            else:
                replace_recursive(child, full_name)
                
    replace_recursive(model)
    return replaced_count


def load_model_for_inference(crom_path: str):
    print(f"⏳ Lendo arquivo {crom_path}...")
    t0 = time.time()
    config, tensors = load_cromv3_sota_file(crom_path)
    
    hf_model_id = config.get("hf_model_id")
    K = config.get("crom_K", 256)
    D = config.get("crom_D", 8)
    
    print(f"🧩 Configurações Crom: K={K}, D={D} | Modelo Original: {hf_model_id}")
    
    print("⏳ Carregando esqueleto HuggingFace vazio (meta-device)...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(hf_model_id)
    
    # Usa o meta-device para inicializar matematicamente em 0.01 segundos
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)
        
    # Materializa a memória VAZIA antes de injetar os kernels
    model.to_empty(device="cpu")
    
    print("⏳ Injetando Kernels C++ CromLinear no esqueleto...")
    count = inject_cromlinear_layers(model, tensors, K, D)
    print(f"✅ {count} camadas substituídas com sucesso!")
    
    print("⏳ Carregando os tensores FP16 restantes (Embeddings, Norms)...")
    
    # Filtra os tensores que não são codebooks ou índices
    hf_tensors = {k: v for k, v in tensors.items() if ".crom_" not in k}
    model.load_state_dict(hf_tensors, strict=False)
    
    t1 = time.time()
    print(f"🚀 Sistema Inicializado em {t1-t0:.2f}s!")
    
    return model, tokenizer

def chat_loop(crom_path: str):
    model, tokenizer = load_model_for_inference(crom_path)
    model.eval()
    
    print("\n" + "="*50)
    print("🎙️ CROM GPT-4 LOCAL (Powered by C++ AVX)")
    print("="*50)
    print("A memória RAM da sua máquina deve estar cravada e suave agora.")
    
    # Lista de mensagens para Chat Template
    messages = [
        {"role": "system", "content": "Você é um assistente brilhante, altamente inteligente e prestativo. Responda sempre em português claro e conciso."}
    ]
    
    # Streamer customizado para forçar o print imediato (ignora buffer de palavras)
    from transformers.generation.streamers import BaseStreamer
    class ImmediateStreamer(BaseStreamer):
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
        def put(self, value):
            if len(value.shape) == 1:
                value = value.unsqueeze(0)
            text = self.tokenizer.decode(value[0], skip_special_tokens=True)
            print(text, end="", flush=True)
        def end(self):
            print()
            
    while True:
        try:
            prompt = input("\n>> Você: ")
            if prompt.lower() in ['sair', 'exit']:
                break
                
            messages.append({"role": "user", "content": prompt})
            
            # Aplica o formato perfeito (Llama-3 ou Phi-3)
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer.encode(input_text, return_tensors='pt')
            
            # Adicionamos attention mask explícito para evitar o warning que você viu
            attention_mask = torch.ones_like(input_ids)
            
            print("\n🤖 Assistente: ", end="", flush=True)
            
            # Usamos nosso streamer que não segura buffers!
            streamer = ImmediateStreamer(tokenizer)
            
            t_start = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    streamer=streamer
                )
            t_end = time.time()
            
            # Corta o input do output
            gen_ids = output_ids[0][input_ids.shape[1]:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)
            
            # Adiciona ao histórico
            messages.append({"role": "assistant", "content": response})
            
            # Calcula velocidade
            num_tokens = len(gen_ids)
            tok_s = num_tokens / (t_end - t_start)
            
            print(f"\n⚡ [Velocidade: {tok_s:.2f} tokens/s | Geração: {t_end - t_start:.2f}s]")
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    if len(sys.argv) > 1:
        crom_path = sys.argv[1]
    else:
        crom_path = "phi3_crom.cromv3"
        
    if not os.path.exists(crom_path):
        print(f"❌ Arquivo {crom_path} não encontrado! Rode o compressor na nuvem primeiro e baixe o arquivo.")
        sys.exit(1)
        
    chat_loop(crom_path)
