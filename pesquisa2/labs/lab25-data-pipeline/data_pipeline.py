"""
Data Pipeline — Coleta, Limpeza e Tokenização de Corpus PT
===========================================================

Pipeline para preparar dados em Português para treinar o CromGPT.

Etapas:
1. Download de datasets PT do HuggingFace
2. Filtragem de qualidade
3. Tokenização
4. Salvar formato pronto para DataLoader

Pesquisa 2 — Lab 25
"""

import os
import json
import hashlib
import argparse
import numpy as np
from pathlib import Path


def download_wikipedia_pt(output_dir: str, max_articles: int = 50000):
    """
    Baixa Wikipedia PT do HuggingFace.
    Usa streaming para não precisar baixar tudo de uma vez.
    """
    from datasets import load_dataset
    
    print(f"📥 Baixando Wikipedia PT (max {max_articles:,} artigos)...")
    
    ds = load_dataset(
        "wikimedia/wikipedia", 
        "20231101.pt",
        split="train",
        streaming=True,
        trust_remote_code=True
    )
    
    texts = []
    seen_hashes = set()
    
    for i, example in enumerate(ds):
        if i >= max_articles:
            break
        
        text = example.get("text", "")
        
        # Filtro de qualidade básico
        if len(text) < 100:  # Artigos muito curtos
            continue
        
        # Deduplicação por hash
        text_hash = hashlib.md5(text[:500].encode()).hexdigest()
        if text_hash in seen_hashes:
            continue
        seen_hashes.add(text_hash)
        
        texts.append(text)
        
        if len(texts) % 5000 == 0:
            print(f"  ... {len(texts):,} artigos coletados")
    
    # Salvar como JSONL
    output_path = os.path.join(output_dir, "wikipedia_pt_raw.jsonl")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for text in texts:
            json.dump({"text": text}, f, ensure_ascii=False)
            f.write("\n")
    
    total_chars = sum(len(t) for t in texts)
    print(f"  ✅ {len(texts):,} artigos salvos")
    print(f"  ✅ {total_chars:,} caracteres totais")
    print(f"  ✅ Arquivo: {output_path}")
    
    return output_path, len(texts), total_chars


def clean_and_filter(input_path: str, output_dir: str, min_len: int = 200):
    """
    Filtra e limpa o corpus:
    - Remove textos muito curtos
    - Remove linhas com marcação wiki residual
    - Normaliza espaços
    """
    print(f"\n🧹 Limpando e filtrando...")
    
    texts = []
    skipped = 0
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            text = doc["text"]
            
            # Limpeza básica
            lines = text.split("\n")
            clean_lines = []
            for l in lines:
                l = l.strip()
                # Pular linhas de marcação wiki
                if l.startswith("==") or l.startswith("{{") or l.startswith("|}"):
                    continue
                if l.startswith("*") and len(l) < 30:  # Listas curtas
                    continue
                if l.startswith("|"):  # Tabelas wiki
                    continue
                if len(l) < 10:
                    continue
                clean_lines.append(l)
            
            clean_text = " ".join(clean_lines)
            
            if len(clean_text) < min_len:
                skipped += 1
                continue
            
            texts.append(clean_text)
    
    output_path = os.path.join(output_dir, "wikipedia_pt_clean.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for text in texts:
            json.dump({"text": text}, f, ensure_ascii=False)
            f.write("\n")
    
    total_chars = sum(len(t) for t in texts)
    print(f"  ✅ {len(texts):,} documentos limpos (descartou {skipped:,})")
    print(f"  ✅ {total_chars:,} caracteres")
    
    return output_path, len(texts)


def tokenize_corpus(input_path: str, output_dir: str, 
                    tokenizer_name: str = "pierreguillou/gpt2-small-portuguese",
                    max_seq_len: int = 512):
    """
    Tokeniza o corpus limpo e salva em formato binário pronto para treino.
    
    Formato de saída: numpy array de tokens concatenados,
    com <|endoftext|> separando documentos.
    """
    from transformers import AutoTokenizer
    
    print(f"\n📝 Tokenizando com {tokenizer_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Se o tokenizer não tem pad token, usar eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    eos_id = tokenizer.eos_token_id
    vocab_size = tokenizer.vocab_size
    
    all_tokens = []
    doc_count = 0
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            text = doc["text"]
            
            # Tokenizar
            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            # Adicionar tokens + EOS separator
            all_tokens.extend(tokens)
            all_tokens.append(eos_id)
            
            doc_count += 1
            if doc_count % 5000 == 0:
                print(f"  ... {doc_count:,} docs tokenizados, {len(all_tokens):,} tokens")
    
    # Converter para numpy
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    
    # Dividir em train/val (95/5)
    n = len(all_tokens)
    split_idx = int(n * 0.95)
    
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]
    
    # Salvar
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.npy")
    val_path = os.path.join(output_dir, "val.npy")
    meta_path = os.path.join(output_dir, "meta.json")
    
    np.save(train_path, train_tokens)
    np.save(val_path, val_tokens)
    
    meta = {
        "tokenizer": tokenizer_name,
        "vocab_size": vocab_size,
        "eos_token_id": eos_id,
        "total_tokens": int(n),
        "train_tokens": int(len(train_tokens)),
        "val_tokens": int(len(val_tokens)),
        "documents": doc_count,
        "max_seq_len": max_seq_len,
    }
    
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n  ✅ Total tokens: {n:,}")
    print(f"  ✅ Train: {len(train_tokens):,} tokens ({train_path})")
    print(f"  ✅ Val: {len(val_tokens):,} tokens ({val_path})")
    print(f"  ✅ Vocab size: {vocab_size:,}")
    print(f"  ✅ Meta: {meta_path}")
    
    return meta


def create_mini_dataset(output_dir: str, n_tokens: int = 100000):
    """
    Cria um mini-dataset sintético para testes rápidos locais.
    Usa texto em PT gerado a partir de templates simples.
    """
    print(f"\n🔧 Criando mini-dataset de teste ({n_tokens:,} tokens)...")
    
    from transformers import AutoTokenizer
    
    tokenizer_name = "pierreguillou/gpt2-small-portuguese"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Textos de exemplo em PT para teste
    templates = [
        "O Brasil é um país localizado na América do Sul. É o maior país da região e possui uma rica diversidade cultural e natural.",
        "A cidade de São Paulo é a maior metrópole do Brasil. Com mais de doze milhões de habitantes, é um importante centro econômico e cultural.",
        "O Rio de Janeiro é famoso por suas praias, como Copacabana e Ipanema. O Cristo Redentor é um dos monumentos mais visitados do mundo.",
        "A Amazônia é a maior floresta tropical do planeta. Ela abriga uma imensa biodiversidade e é fundamental para o equilíbrio climático global.",
        "A culinária brasileira é rica e variada. Pratos como feijoada, acarajé e pão de queijo são conhecidos em todo o mundo.",
        "O futebol é o esporte mais popular do Brasil. A seleção brasileira é a maior vencedora da Copa do Mundo, com cinco títulos.",
        "A educação no Brasil tem avançado nos últimos anos. O acesso ao ensino superior tem aumentado, embora desafios persistam.",
        "A música brasileira é conhecida mundialmente. Gêneros como samba, bossa nova e forró fazem parte da identidade cultural do país.",
        "A tecnologia tem transformado o cotidiano dos brasileiros. O uso de smartphones e internet cresce a cada ano no país.",
        "O Pantanal é a maior área úmida do mundo. Localizado nos estados de Mato Grosso e Mato Grosso do Sul, abriga uma fauna diversificada.",
        "A inteligência artificial está revolucionando diversos setores. Desde a medicina até a agricultura, algoritmos inteligentes otimizam processos.",
        "A programação de computadores é uma habilidade essencial no mundo moderno. Linguagens como Python, JavaScript e Go são muito utilizadas.",
        "O sistema solar é composto por oito planetas que orbitam o Sol. A Terra é o único planeta conhecido que abriga vida.",
        "A história do Brasil começou oficialmente em mil e quinhentos. Os portugueses chegaram ao litoral brasileiro e iniciaram a colonização.",
        "O carnaval é a festa mais popular do Brasil. Com desfiles de escolas de samba e blocos de rua, atrai milhões de turistas.",
    ]
    
    # Repetir e variar para gerar volume
    all_tokens = []
    eos_id = tokenizer.eos_token_id
    
    i = 0
    while len(all_tokens) < n_tokens:
        text = templates[i % len(templates)]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
        all_tokens.append(eos_id)
        i += 1
    
    all_tokens = np.array(all_tokens[:n_tokens], dtype=np.uint16)
    
    # Split
    split = int(len(all_tokens) * 0.9)
    train = all_tokens[:split]
    val = all_tokens[split:]
    
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "train.npy"), train)
    np.save(os.path.join(output_dir, "val.npy"), val)
    
    meta = {
        "tokenizer": tokenizer_name,
        "vocab_size": tokenizer.vocab_size,
        "eos_token_id": eos_id,
        "total_tokens": len(all_tokens),
        "train_tokens": len(train),
        "val_tokens": len(val),
        "documents": i,
        "max_seq_len": 512,
        "type": "mini_synthetic",
    }
    
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"  ✅ {len(all_tokens):,} tokens gerados ({i} documentos)")
    print(f"  ✅ Train: {len(train):,} | Val: {len(val):,}")
    print(f"  ✅ Vocab: {tokenizer.vocab_size:,}")
    
    return meta


# ═══════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Pipeline para CromGPT")
    parser.add_argument("--mode", choices=["mini", "wiki", "full"], default="mini",
                       help="mini: dataset sintético rápido | wiki: Wikipedia PT | full: pipeline completo")
    parser.add_argument("--output", default="../../data", help="Diretório de saída")
    parser.add_argument("--max-articles", type=int, default=50000, help="Máx artigos Wikipedia")
    parser.add_argument("--mini-tokens", type=int, default=100000, help="Tokens para mini dataset")
    
    args = parser.parse_args()
    
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  DATA PIPELINE — Corpus PT para CromGPT                 ║")
    print("║  Pesquisa 2: CromGPT (LLM Nativo .crom)                ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    
    if args.mode == "mini":
        meta = create_mini_dataset(args.output, n_tokens=args.mini_tokens)
    
    elif args.mode == "wiki":
        raw_path, n_docs, n_chars = download_wikipedia_pt(args.output, args.max_articles)
        clean_path, n_clean = clean_and_filter(raw_path, args.output)
        meta = tokenize_corpus(clean_path, args.output)
    
    elif args.mode == "full":
        raw_path, n_docs, n_chars = download_wikipedia_pt(args.output, args.max_articles)
        clean_path, n_clean = clean_and_filter(raw_path, args.output)
        meta = tokenize_corpus(clean_path, args.output)
    
    print(f"\n📊 Metadados finais:")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print()
