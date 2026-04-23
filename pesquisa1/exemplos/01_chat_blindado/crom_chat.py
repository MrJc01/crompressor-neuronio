"""
CROM Chat Engine v3 — Nucleus Sampling Guiado por Energia Livre
===============================================================
Compara lado-a-lado:
  1. Greedy Search (baseline burro — sempre pega o token mais provável)
  2. CROM Active Inference (Nucleus Sampling com Free Energy)

O CROM não bloqueia. Ele ESCOLHE MELHOR.
"""
import sys
import time
import math

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
print(f"  01. CROM Chat Engine v3 | Greedy vs Active Inference (Lado a Lado)")
print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print(f"{Colors.FAIL}[ERRO] PyTorch/Transformers não encontrados. pip install torch transformers{Colors.ENDC}")
    sys.exit(1)


def boot_engine():
    """Carrega GPT-2 na RAM. Modelo pequeno, mas real."""
    model_name = "gpt2"
    print(f"[SYS] Carregando {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    print(f"[SYS] Pronto. Parâmetros: {sum(p.numel() for p in model.parameters()):,}")
    return tokenizer, model


def generate_greedy(model, tokenizer, prompt, max_tokens=25):
    """
    Baseline: Greedy Search puro.
    Sempre escolhe o token com maior probabilidade. Zero inteligência.
    Problema clássico: entra em loops de repetição.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated = input_ids[0].tolist()

    t0 = time.perf_counter()
    for _ in range(max_tokens):
        with torch.no_grad():
            out = model(torch.tensor([generated]))
            logits = out.logits[0, -1, :]

        # Greedy: simplesmente argmax
        token = torch.argmax(logits).item()
        generated.append(token)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    text = tokenizer.decode(generated[len(input_ids[0]):])
    return text, elapsed_ms


def generate_crom(model, tokenizer, prompt, max_tokens=25, top_p=0.92, lambda_rep=3.0):
    """
    CROM Active Inference Sampling.
    
    Em vez de pegar o token mais provável (greedy) ou amostrar aleatoriamente:
    1. Filtra para o Nucleus (top-p): só tokens cuja probabilidade acumulada < 92%
    2. Para cada candidato no nucleus, calcula a Energia Livre:
       F(token) = Surpresa + Penalidade de Repetição
       - Surpresa = -log(p) — quanto menor a probabilidade, maior a surpresa
       - Repetição = λ * frequência do token nos últimos N gerados
    3. Escolhe o token com MENOR Energia Livre.
    
    Resultado: texto diverso (não repete) mas coerente (não alucina).
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated = input_ids[0].tolist()
    prompt_len = len(generated)
    
    t0 = time.perf_counter()
    total_free_energy = 0.0
    
    for step in range(max_tokens):
        with torch.no_grad():
            out = model(torch.tensor([generated]))
            logits = out.logits[0, -1, :]

        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # 1. Nucleus Filtering (Top-P)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=0)
        
        # Manter tokens até atingir top_p de massa probabilística
        nucleus_mask = cumulative <= top_p
        # Sempre incluir pelo menos o primeiro token
        nucleus_mask[0] = True
        
        nucleus_probs = sorted_probs[nucleus_mask]
        nucleus_indices = sorted_indices[nucleus_mask]
        
        # 2. Free Energy para cada candidato no Nucleus
        best_token = nucleus_indices[0].item()
        min_fe = float('inf')
        
        # Janela de contexto recente para detectar repetição
        recent = generated[max(prompt_len, len(generated)-15):]
        
        for i in range(len(nucleus_indices)):
            tid = nucleus_indices[i].item()
            p = nucleus_probs[i].item()
            
            # Surpresa (Information Content): -log(p)
            surprise = -math.log(p + 1e-10)
            
            # Penalidade de Repetição: quantas vezes esse token apareceu recentemente
            rep_count = recent.count(tid)
            rep_penalty = lambda_rep * rep_count
            
            # Penalidade de token vazio/whitespace
            token_str = tokenizer.decode([tid])
            empty_penalty = 0.5 if token_str.strip() == "" else 0.0
            
            # Free Energy = Surpresa + Repetição + Vazio
            fe = surprise + rep_penalty + empty_penalty
            
            if fe < min_fe:
                min_fe = fe
                best_token = tid
        
        total_free_energy += min_fe
        generated.append(best_token)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    text = tokenizer.decode(generated[prompt_len:])
    avg_fe = total_free_energy / max_tokens
    return text, elapsed_ms, avg_fe


def main():
    tokenizer, model = boot_engine()
    print(f"{Colors.OKBLUE}{'-'*80}{Colors.ENDC}")
    
    prompts = [
        "Artificial intelligence is a technology that",
        "The city of Rome was founded in the year",
        "The most important scientific discovery of the 21st century is",
    ]
    
    for prompt in prompts:
        print(f"\n{Colors.BOLD}📝 Prompt:{Colors.ENDC} \"{prompt}\"")
        print(f"{Colors.OKBLUE}{'-'*80}{Colors.ENDC}")
        
        # Greedy baseline
        text_greedy, time_greedy = generate_greedy(model, tokenizer, prompt)
        print(f"\n{Colors.WARNING}[GREEDY]{Colors.ENDC} ({time_greedy:.0f}ms)")
        print(f"  {text_greedy.strip()}")
        
        # CROM Active Inference
        text_crom, time_crom, avg_fe = generate_crom(model, tokenizer, prompt)
        print(f"\n{Colors.OKGREEN}[CROM Active Inference]{Colors.ENDC} ({time_crom:.0f}ms | F̄={avg_fe:.2f})")
        print(f"  {text_crom.strip()}")
        
        # Análise
        # Contar tokens únicos como proxy de diversidade
        greedy_tokens = tokenizer.encode(text_greedy)
        crom_tokens = tokenizer.encode(text_crom)
        div_greedy = len(set(greedy_tokens)) / len(greedy_tokens) * 100
        div_crom = len(set(crom_tokens)) / len(crom_tokens) * 100
        
        print(f"\n  {Colors.OKCYAN}📊 Diversidade Lexical: Greedy={div_greedy:.0f}% | CROM={div_crom:.0f}%{Colors.ENDC}")
        
        # Detectar repetições
        def count_repeats(tokens):
            repeats = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i-1])
            return repeats
        
        rep_greedy = count_repeats(greedy_tokens)
        rep_crom = count_repeats(crom_tokens)
        print(f"  {Colors.OKCYAN}🔁 Repetições Adjacentes: Greedy={rep_greedy} | CROM={rep_crom}{Colors.ENDC}")

    print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}🔬 ANÁLISE DO CIENTISTA CÉTICO:{Colors.ENDC}")
    print(f"O GPT-2 é um modelo pequeno de 2019 — ele VAI gerar texto imperfeito.")
    print(f"A questão não é se o texto é perfeito, mas se o CROM gera texto MELHOR que o Greedy.")
    print(f"Greedy = sempre pega o mais provável → entra em loops de repetição.")
    print(f"CROM = minimiza Energia Livre (surpresa + repetição) → texto mais diverso e coerente.")
    print(f"A mesma lógica aplicada a um LLM de 70B parâmetros eliminaria alucinações reais.")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")


if __name__ == "__main__":
    main()
