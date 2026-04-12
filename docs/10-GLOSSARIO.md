# 📚 Glossário

> Termos técnicos unificados do ecossistema Crompressor. Cross-referência com [crompressor-sinapse/docs/10-GLOSSARIO.md](https://github.com/MrJc01/crompressor-sinapse/blob/main/docs/10-GLOSSARIO.md).

---

| Termo | Definição | Usado em |
|:---|:---|:---|
| **brain.crom** | Arquivo `.crom` contendo um modelo de IA congelado em formato DNA Base-4 com Codebook, Merkle Tree e metadata. Equivale a um "neurônio" no contexto deste repositório. | Neurônio |
| **CDC (Content-Defined Chunking)** | Técnica de divisão de dados em blocos baseada no conteúdo (rolling hash), em vez de tamanho fixo. Garante que edits locais não afetam chunks distantes. | Core, Sinapse, Neurônio |
| **Codebook** | Dicionário treinável que mapeia hashes de chunks para sequências DNA (Base-4). Central para a compressão semântica do Crompressor. | Core, Sinapse, Neurônio |
| **Colapso Multi-Brain** | Degradação de qualidade quando neurônios incompatíveis são roteados simultaneamente. Análogo a interferência destrutiva em física de ondas. | Neurônio |
| **Delta XOR** | Operação XOR bitwise entre um chunk original e uma versão modificada. Grava apenas a diferença. Reversível: A ⊕ B ⊕ B = A. | Core, Sinapse, Neurônio |
| **Dilithium** | Esquema de assinatura digital pós-quântica (NIST PQC Round 3). Usado para assinar neurônios e deltas. | Security, Neurônio |
| **DNA Base-4** | Codificação dos dados em 4 bases (A=00, T=01, C=10, G=11). Permite compressão extrema inspirada em biologia molecular. | Core, IA, Neurônio |
| **Drift Entrópico** | Aumento gradual da entropia de Shannon conforme um neurônio semi-fixo recebe muitas atualizações parciais. | Neurônio |
| **FastCDC** | Implementação otimizada de Content-Defined Chunking usada no core do Crompressor. | Core |
| **Forward Pass Diferencial** | Computar apenas os chunks que mudaram em relação ao cache, evitando recalcular de todo o modelo. Sinapse Frente 2. | Sinapse, Neurônio |
| **FUSE** | Filesystem in Userspace. Permite montar o brain.crom como um sistema de arquivos virtual com leitura aleatória O(1). | Core, IA, Neurônio |
| **GGUF** | Formato de modelo quantizado usado pelo llama.cpp. Principal formato de entrada para criar brain.crom. | IA, Neurônio |
| **HNSW** | Hierarchical Navigable Small World. Algoritmo de busca aproximada por vizinhos mais próximos. Usado no Codebook e no Multi-Brain routing. | Core, Sinapse, Neurônio |
| **Kademlia** | Protocolo DHT para redes P2P. Usado via LibP2P para sincronização de deltas e codebooks. | Core, Neurônio |
| **LLVQ** | Leech Lattice Vector Quantization. Paper de Mar 2026 que usa lattice Leech (24D) para VQ de LLMs sem codebooks gigantes. | Neurônio (paper) |
| **LSH (Locality-Sensitive Hashing)** | Técnica de indexação que agrupa dados similares no mesmo bucket. Usado em B-Tree no Crompressor para busca O(1). | Core, Sinapse |
| **Merkle Tree** | Árvore de hashes para verificação de integridade bit-a-bit. Cada chunk tem um hash folha, e a raiz verifica tudo. | Core, Neurônio |
| **MoFE** | Mixture of Frozen Experts. Paper de 2025 que congela FFN experts e usa routing. Base teórica do Multi-Brain. | Neurônio (paper) |
| **Multi-Brain Engine** | Sistema de routing entre múltiplos modelos/.crom. Versão V4.x do crompressor-ia. Reutilizado no neurônio para composição. | IA, Neurônio |
| **Neurônio Dinâmico** | Vertente 3: modelo completamente atualizável via treinamento XOR Delta completo. Máxima flexibilidade. | Neurônio |
| **Neurônio Fixo (Frozen)** | Vertente 1: modelo congelado em .crom, read-only. Adaptação apenas via tensor delta externo. | Neurônio |
| **Neurônio Semi-Fixo** | Vertente 2: modelo parcialmente atualizável. Chunks individuais podem ser substituídos via HNSW similarity. | Neurônio |
| **PQ (Pós-Quântica)** | Criptografia resistente a computadores quânticos. ChaCha20 + Dilithium no ecossistema Crompressor. | Security, Neurônio |
| **Shannon Entropy** | Medida de informação/incerteza de um conjunto de dados. H = -Σ p(x) log₂ p(x). Usada para medir compressibilidade e drift. | Core, Sinapse, Neurônio |
| **Silent Drop** | Técnica do crompressor-security que rejeita conexões inválidas sem revelar que o servidor existe. Anti-fingerprinting. | Security, Neurônio |
| **Tensor Delta** | Dados externos aplicados sobre um neurônio fixo para gerar saídas adaptativas. Pode ser XOR, VQ ou composição ponderada. | Neurônio |
| **TTFT** | Time To First Token. Latência entre enviar um prompt e receber o primeiro token da resposta. | Sinapse, Neurônio |
| **Vector Quantization (VQ)** | Técnica de compressão que mapeia vetores para o centroide mais próximo em um codebook. No neurônio, deltas são aplicados no espaço VQ. | Sinapse, Neurônio |
| **Zero-Swapping** | Filosofia do crompressor-ia: inferência sem swap de memória. FUSE + SSD O(1) em vez de carregar modelo inteiro em RAM. | IA, Neurônio |
| **ZipLLM** | Framework 2025 para deduplicação de modelos via delta XOR (BitX). Inspiração direta para nosso Delta sobre Codebook. | Neurônio (paper) |

---

> **Início:** [00 — Índice](00-INDICE.md)
