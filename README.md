<div align="center">

![Version](https://img.shields.io/badge/Version-V0.2_Experimental-9d4edd)
![Go](https://img.shields.io/badge/Core-Go_1.22+-00ADD8)
![Python](https://img.shields.io/badge/Research-Python_3.12_+_PyTorch-3776AB)
![Status](https://img.shields.io/badge/Status-Resultados_Positivos-00ff88)
![License](https://img.shields.io/badge/License-MIT-orange)

# 🧬 Crompressor-Neurônio

**"O neurônio que comprime é o neurônio que pensa."**

*Laboratório de pesquisa para explorar o uso do motor [Crompressor](https://github.com/MrJc01/crompressor) como substituto de representações neurais — desde pesos de tensores individuais até modelos LLM completos.*

</div>

---

## 📊 Status Real do Projeto

| Componente | O Que Funciona | O Que é PoC/Stub |
|:---|:---|:---|
| **Tensor-Vivo** (pesquisa ativa) | ✅ Codebook Learning: 97.56% acc com 40.8x compressão | CDC hash exato (não encontra dedup) |
| **Pesquisa0** (5D Neural) | ✅ 12/12 labs, 103/124 items, GPU validado | Migração Go pendente |
| **Cérebro-FUSE** (pausado) | ✅ FUSE driver real, FastCDC sobre GGUF, UI cockpit | Formato .crom binário, FUSE lazy read |
| **Testes Go** | ✅ 17 testes + 5 benchmarks (Shannon, XOR, Merkle) | VQ Delta (fake), "HNSW" (é brute-force) |
| **Segurança** | ✅ Ed25519 sign/seal funcional | Dilithium PQC (placeholder) |

---

## 🔬 Duas Linhas de Pesquisa

### 🧠 Tensor-Vivo — *Crompressor como substituto de tensores* `[ATIVO]`

> **Pergunta:** E se, em vez de floats, o "peso" de cada neurônio fosse o sentido comprimido pelo Crompressor?

**Resultados empíricos (MNIST MLP, 235K params):**

| Experimento | Resultado |
|:---|:---|
| **Exp0:** CDC sobre pesos reais | 0% dedup (hash exato não serve para pesos float) |
| **Exp1:** Codebook K-Means → Accuracy | K=128: **96.43%** acc com **18.5x** compressão |
| **Exp2:** Treinar APENAS o codebook | K=128: **97.56%** acc com **5,770 params** (40.8x menos) |
| | K=256 B=32: **98.08%** — **SUPEROU o baseline** (97.53%) |

> **Veredicto:** O Codebook do Crompressor **é um espaço de aprendizado viável**. Funciona como um "LoRA no espaço comprimido".

### 🔬 Pesquisa0 — *Motor 5D de Inferência Ativa* `[ATIVO — 83% completo]`

> **Pergunta:** E se o Crompressor fosse o motor de simulação de um agente que pensa em branches de futuro?

**12 laboratórios experimentais validados:**

| Lab | Resultado Principal |
|:---|:---|
| Lab01 — FPS Computacional | IA opera a 14.303x vs humano |
| Lab03 — World Model | Erro < 5%, Energia Livre converge |
| Lab06 — KV Cache Codebook | **94.2% redução real (GPT-2, Tesla T4)** |
| Lab07 — Delta Branches | 99.9% economia de memória |
| Lab10 — Active Inference | 12.7x speedup sobre random |

> **Score:** 13 hipóteses confirmadas, 1 parcial, 0 refutadas

```
pesquisa0/
├── PLANEJAMENTO.md        # Checklist 124 itens (103 completos)
├── CONCLUSOES.md          # Veredicto final
├── labs/                  # 12 laboratórios + 3 blitz scripts
├── papers/                # papel0.md, papel1.md, papel2.md, papel3.md
└── resultados/            # JSONs de todos os experimentos
```

### ⚡ Cérebro-FUSE — *LLMs inteiros como neurônios via FUSE* `[PAUSADO]`

> **Ideia:** Congelar modelos LLM completos como neurônios `.crom` e servi-los via FUSE kernel driver com deltas XOR compostos.

```
pesquisas/cerebro-fuse/
├── PLANEJAMENTO.md     # Checklist completo (6 fases, ~80 itens)
└── README.md           # Descrição da linha de pesquisa
```

Código existente em `pesquisas/testes/` e `web-cockpit/`.

---

## 📂 Estrutura do Repositório

```
crompressor-neuronio/
├── README.md                          # ← Você está aqui
├── docs/                              # 12 docs técnicos (manifesto → glossário)
├── pesquisas/
│   ├── tensor-vivo/                   # 🔬 Pesquisa ativa: codebook como tensor
│   │   ├── exp0_analise_estrutural/   #    CDC + entropia sobre pesos
│   │   ├── exp1_roundtrip/            #    K-Means quantization
│   │   ├── exp2_codebook_learning/    #    CodebookLinear + training
│   │   ├── dados/                     #    JSONs de resultados
│   │   ├── CONCLUSOES.md              #    Veredicto
│   │   └── PLANEJAMENTO.md            #    Checklist extenso
│   ├── cerebro-fuse/                  # ⏸️ LLMs como neurônios via FUSE
│   ├── testes/                        # Go: neuronio.go, FUSE, routing, API
│   │   ├── cmd/                       #    Binários de teste
│   │   └── pkg/                       #    Bibliotecas (fuse_mount, routing, etc.)
│   ├── papers/                        # Referências acadêmicas
│   ├── scripts/                       # Automação
│   └── visualizacao/                  # Python dashboards
├── web-cockpit/                       # React UI (Canvas pipeline graph)
└── LICENSE                            # MIT
```

---

## 🚀 Quick Start

### Tensor-Vivo (Python)
```bash
cd pesquisas/tensor-vivo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Treinar modelo MNIST
python exp0_analise_estrutural/train_mnist.py

# Testar codebook quantization
python exp1_roundtrip/codebook_quantize.py

# Treinar apenas o codebook (resultado principal)
python exp2_codebook_learning/train_codebook.py
```

### Cérebro-FUSE (Go)
```bash
cd pesquisas/testes
go test ./pkg/...          # 17 testes unitários
go run cmd/run_all/main.go # Suite completa
```

### Web Cockpit (React)
```bash
cd web-cockpit
npm install && npm run dev
```

---

## 🔬 Pesquisa Acadêmica Base

| Área | Paper/Conceito | Relevância |
|:---|:---|:---|
| Vector Quantization | **FVQ/VQBridge** (2025) | Codebooks com 100% utilização |
| Codebook Features | **ICML 2025** | Hidden states quantizados = interpretabilidade |
| Low-Dim Codebook | **LooC** (2026) | VQ em slots sub-vetor (similar a chunking) |
| Delta Adapters | **FLoRA / aLoRA** (2025) | Adapters fundidos em modelo frozen |
| Model Dedup | **ZipLLM** + **BitX** (2025) | Delta XOR entre modelos |
| SSD Inference | **FlexInfer / HILOS** (2025-2026) | Near-storage processing |
| Termodinâmica | **arXiv:2512.02221** (2025) | Shannon ↔ Boltzmann em redes neurais |

---

## 🤝 Ecossistema Completo

| Repositório | Foco |
|:---|:---|
| [**crompressor**](https://github.com/MrJc01/crompressor) | Motor semântico core (CDC, Codebook, FUSE, Merkle) |
| [**crompressor-projetos**](https://github.com/MrJc01/crompressor-projetos) | 25 labs WASM no browser |
| [**crompressor-ia**](https://github.com/MrJc01/crompressor-ia) | IA Sub-Simbólica Edge |
| [**crompressor-security**](https://github.com/MrJc01/crompressor-security) | Segurança pós-quântica |
| [**crompressor-sinapse**](https://github.com/MrJc01/crompressor-sinapse) | Pesquisa: compressão = cognição |
| **crompressor-neuronio** | **Este repo: laboratório do neurônio** |

---

<div align="center">

*"Nós não comprimimos dados. Nós indexamos o universo."*

**Autor:** [MrJc01](https://github.com/MrJc01) · **Licença:** MIT · **Data:** Abril de 2026

</div>