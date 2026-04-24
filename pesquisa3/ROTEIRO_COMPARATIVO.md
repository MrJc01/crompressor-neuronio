# Roteiro Comparativo: Pesquisa 3

**Data:** 2026-04-24  
**Objetivo:** Solucionar a falha de geração (Semantic Drift) e a lentidão (Overhead de C++) encontradas na primeira iteração de Post-Training Quantization (PTQ) da Pesquisa 3.

## A Decisão Estratégica
Em vez de focar cegamente em um único caminho, a decisão estratégica foi adotar uma abordagem de teste A/B para documentar cientificamente o impacto do PTQ parcial vs QAT (Quantization Aware Training).

### 1. Testar o Caminho B (PTQ Híbrido Python)
**Execução Imediata (Ambiente Local - 8GB RAM)**
- **Como Funciona:** Faremos PTQ apenas nas camadas do Feed Forward Network (FFN), que representam ~60% do tamanho do modelo e são menos sensíveis à compressão bruta. As camadas de Atenção (Q, K, V) e a boca do modelo (`lm_head`) serão preservadas intocadas em precisão total (FP16).
- **Abandono do C++:** O C++ puro não consegue competir com os Kernels OneBLAS que o PyTorch já utiliza para multiplicação de matrizes densas. Retornaremos à arquitetura do *Papel 2*, reconstruindo a matriz quantizada em tempo real no Python (`W = codebook[indices]`) e injetando no otimizador nativo do PyTorch, alcançando uma velocidade interativa.
- **Resultado Esperado:** O modelo `TinyLlama` recuperará 90% do seu QI, rodará de forma muito rápida e ainda ocupará menos memória RAM, viabilizando o chat local.

### 2. Testar o Caminho A (QAT Nativo Nuvem)
**Execução Futura (Ambiente Nuvem - Vast.ai)**
- **Como Funciona:** Voltar à tese de ouro da Pesquisa 2. Em vez de esmagar o Llama depois de pronto, continuaremos o treinamento de um modelo CromGPT nativo onde 100% das camadas (incluindo Atenção e FFN) são Codebooks que aprendem de forma simbiótica com o Straight-Through Estimator.
- **Resultado Esperado:** Um modelo puro, sem perda qualitativa posterior e 100% comprimido (diferente do caminho B que é parcialmente comprimido).

## Métrica de Sucesso
O vencedor entre o Caminho B e o Caminho A ditará a arquitetura final da engine de inferência do Crompressor para modelos de bilhões de parâmetros na borda (Edge Devices).
