<div align="center">

![Version](https://img.shields.io/badge/Version-V0.1_Exploratório-9d4edd)
![Go](https://img.shields.io/badge/Core-Go_1.22+-00ADD8)
![Status](https://img.shields.io/badge/Status-Pesquisa_%26_Desenvolvimento-00ff88)
![License](https://img.shields.io/badge/License-MIT-orange)

# 🧬 Crompressor-Neurônio

**"O neurônio que comprime é o neurônio que pensa."**

*Laboratório de pesquisa para explorar o uso de modelos completos como neurônios pré-treinados fixos (ou semi-fixos) gerando saídas não-determinísticas via tensores delta — construído sobre o motor [Crompressor](https://github.com/MrJc01/crompressor).*

</div>

---

## 📖 O Que É Isso?

O **crompressor-sinapse** perguntou: *"Compressão é cognição?"*
O **crompressor-neuronio** responde: *"Sim — e aqui está o neurônio operacional."*

Este repositório é um **laboratório de exploração** onde tratamos um modelo de IA completo (o "cérebro") como um **neurônio pré-treinado armazenado no formato `.crom`** (DNA Base-4 + Codebook + Merkle Tree). A partir desse neurônio, geramos saídas adaptativas aplicando **tensores delta** (XOR, Vector Quantization, Multi-Brain Routing) — tudo mantendo leitura fractal O(1) diretamente do SSD via FUSE.

### Três Vertentes de Pesquisa

| Vertente | Cérebro | Geração Não-Fixa | Hipótese |
|:---|:---|:---|:---|
| 🔒 **Neurônio Fixo (Frozen)** | Imutável `.crom` | Tensores delta XOR sobre codebook | Zero training em runtime, compressão extrema |
| 🔄 **Neurônio Semi-Fixo** | CDC chunks parciais | Atualiza frações do codebook via HNSW | Adaptação contínua sem retraining completo |
| ⚡ **Neurônio Dinâmico** | Atualizável via P2P | Treinamento XOR Delta completo | Máxima flexibilidade para edge personalizado |

> **Recomendação:** Exploraremos TODAS as vertentes simultaneamente, gerando dados comparativos para análise.

---

## 🏗️ Arquitetura no Ecossistema

```
                    ┌──────────────────┐
                    │   crompressor    │ ◄── Motor Core (CDC, Codebook, FUSE, Merkle)
                    │      (Core)      │
                    └────────┬─────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
    ┌─────┴──────┐    ┌──────┴──────┐    ┌──────┴──────┐
    │ crompressor │    │ crompressor │    │ crompressor │
    │   -sinapse  │    │    -ia      │    │  -security  │
    │  (Pesquisa) │    │ (Edge IA)   │    │ (Segurança) │
    └─────┬──────┘    └──────┬──────┘    └──────┬──────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
                    ┌────────┴─────────┐
                    │   crompressor-   │ ◄── ESTE REPOSITÓRIO
                    │    neurônio      │
                    │  (Lab Prático)   │
                    └──────────────────┘
```

---

## 🚀 Fluxo de Operação

```
1. TRAIN-BRAIN     crompressor train --dna --domain=brain --input=model.gguf
                   → Gera brain.crom (DNA Base-4 + Codebook)

2. FREEZE          Marca .crom como read-only (flag no header Merkle)
                   → Neurônio fixo, imutável, verificável

3. INFER-DELTA     Carrega tensores delta via FUSE → forward pass diferencial
                   → Saída adaptativa sem modificar o cérebro

4. ROUTE/UPDATE    Multi-brain routing OU delta permanente
                   → Criatividade emergente via composição de neurônios
```

---

## 📂 Estrutura do Repositório

```
crompressor-neuronio/
├── README.md                      # ← Você está aqui
├── docs/                          # Documentação técnica completa
│   ├── 00-INDICE.md               # Índice de navegação
│   ├── 01-MANIFESTO.md            # Tese: compressão → neurônio → cognição
│   ├── 02-TRES-VERTENTES.md       # Fixo vs Semi-Fixo vs Dinâmico
│   ├── 03-ARQUITETURA.md          # Diagramas técnicos detalhados
│   ├── 04-TENSOR-DELTA.md         # XOR Delta + Vector Quantization
│   ├── 05-INTEGRACAO-ECOSSISTEMA.md # Mapa de dependências
│   ├── 06-CASOS-DE-USO.md         # Cenários práticos
│   ├── 07-SEGURANCA-SOBERANIA.md  # Criptografia pós-quântica
│   ├── 08-BENCHMARKS-ESPERADOS.md # Hipóteses e métricas
│   ├── 09-ROADMAP.md              # Fases de pesquisa
│   └── 10-GLOSSARIO.md            # Termos unificados
├── pesquisas/                     # ← LABORATÓRIO DE DADOS
│   ├── papers/                    # Referências acadêmicas (2025-2026)
│   │   └── REFERENCIAS.md         # Catálogo de papers organizados
│   ├── testes/                    # Scripts Go para gerar dados
│   │   ├── cmd/                   # Binários de teste
│   │   ├── pkg/                   # Bibliotecas compartilhadas
│   │   └── Makefile               # Build automatizado
│   ├── scripts/                   # Scripts .sh para automação
│   │   ├── run_all_tests.sh       # Executa todos os testes
│   │   ├── generate_report.sh     # Gera relatórios agregados
│   │   └── benchmark.sh           # Benchmarks comparativos
│   ├── visualizacao/              # Python para visualização
│   │   ├── requirements.txt       # Dependências Python
│   │   ├── visualizar_resultados.py
│   │   └── dashboard.py           # Dashboard interativo
│   ├── relatorios/                # Relatórios gerados (.json/.csv/.md)
│   │   └── .gitkeep
│   └── dados/                     # Dados brutos gerados pelos testes
│       └── .gitkeep
└── LICENSE                        # MIT
```

---

## 🔬 Pesquisa Acadêmica Base

Este projeto se apoia em pesquisa de ponta (2025-2026):

| Área | Paper/Conceito | Relevância para Neurônio |
|:---|:---|:---|
| Vector Quantization | **LLVQ** (Leech Lattice VQ, Mar 2026) | VQ sem codebooks gigantes via lattice 24D |
| KV Cache | **TurboQuant** (ICLR 2026) | Compressão 6x do cache de atenção |
| Delta Adapters | **FLoRA / aLoRA** (2025) | Adapters fundidos em modelo frozen |
| Model Dedup | **ZipLLM** + **BitX** (2025) | Delta XOR entre modelos da mesma família |
| MoE Frozen | **MoFE / Brainstacks** (2025-2026) | FFN frozen + routing entre experts |
| SSD Inference | **FlexInfer / HILOS** (2025-2026) | Near-storage processing para LLMs |
| Termodinâmica | **arXiv:2512.02221** (Dec 2025) | Shannon ↔ Boltzmann em redes neurais |
| LLM Thermodynamics | **arXiv:2505.10571** (May 2025) | Leis termodinâmicas no treinamento de LLMs |

> Ver catálogo completo em [`pesquisas/papers/REFERENCIAS.md`](pesquisas/papers/REFERENCIAS.md)

---

## 🛠️ Como Rodar os Testes

```bash
# 1. Rodar todos os testes Go e gerar dados
cd pesquisas
./scripts/run_all_tests.sh

# 2. Gerar relatórios agregados
./scripts/generate_report.sh

# 3. Visualizar resultados (Python)
cd visualizacao
pip install -r requirements.txt
python visualizar_resultados.py

# 4. Dashboard interativo
python dashboard.py
```

---

## 🤝 Ecossistema Completo

| Repositório | Foco | Link |
|:---|:---|:---|
| **crompressor** | Motor semântico core | [GitHub](https://github.com/MrJc01/crompressor) |
| **crompressor-projetos** | 25 labs WASM no browser | [GitHub](https://github.com/MrJc01/crompressor-projetos) |
| **crompressor-ia** | IA Sub-Simbólica Edge | [GitHub](https://github.com/MrJc01/crompressor-ia) |
| **crompressor-security** | Segurança pós-quântica | [GitHub](https://github.com/MrJc01/crompressor-security) |
| **crompressor-sinapse** | Pesquisa: compressão = cognição | [GitHub](https://github.com/MrJc01/crompressor-sinapse) |
| **crompressor-neuronio** | **Este repo: neurônio operacional** | Você está aqui |

---

<div align="center">

*"Nós não comprimimos dados. Nós indexamos o universo."*

**Autor:** [MrJc01](https://github.com/MrJc01) · **Licença:** MIT · **Data:** Abril de 2026

</div>