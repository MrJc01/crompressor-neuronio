# CommVQ & RoPE: A Teoria da Comutatividade no KV Cache

## O Problema: RoPE destrói K-Means

A maioria dos LLMs modernos (LLaMA, Mistral) utiliza **Rotary Position Embedding (RoPE)**. Em vez de adicionar a posição absoluta aos tokens de input, o RoPE multiplica os tensores de Key e Query por uma matriz de rotação que depende estritamente do índice de posição $m$.

Quando tentamos comprimir o KV Cache usando Vector Quantization (VQ) com Distância Euclidiana, encontramos um obstáculo catastrófico:
**O mesmo exato conceito semântico (ex: a palavra "rei") terá vetores completamente diferentes no cache dependendo de estar na posição 5 ou na posição 400.** 

Se aplicarmos o K-Means padrão ao KV Cache (como fizemos no Lab14), o dicionário de centroides vai desperdiçar espaço alocando um cluster para "rei na posição 5", outro para "rei na posição 10", etc. Isso diminui a eficiência geométrica da compressão de contexto.

## A Solução: Commutative Vector Quantization (CommVQ)

Segundo publicações recentes sobre compressão extrema de cache (ex: Apple *CommVQ* 2024 e *KVTC*), precisamos que a quantização **comute** com o operador de rotação. Em linguagem matemática:

$$ VQ(RoPE_m(x)) = RoPE_m(VQ(x)) $$

Se atingirmos essa comutatividade, a Distância Euclidiana durante a predição de classe do VQ será **invariante à rotação temporal**. 

### Como Implementar?

O RoPE age no subespaço de features de duas em duas dimensões (tratando pares de tensores como números complexos $re + im \cdot j$). Como a rotação de números complexos não altera seu Módulo (magnitude), apenas sua Fase (ângulo), a distância entre dois tensores no plano rotacional só é preservada se **desfizermos a rotação** antes do K-Means.

**Pipeline de Escrita no CROM Cache:**
1. Extraímos o KV Cache tensor $K_{cache}$.
2. Para cada token em posição $m$, aplicamos a matriz Inversa de RoPE: $K_{unrot} = RoPE_{-m}(K_{cache})$.
3. Agora $K_{unrot}$ depende puramente de semântica (a posição foi removida temporalmente).
4. Treinamos/Aplicamos o VQ em $K_{unrot} \rightarrow Indices$.
5. Guardamos os Índices (1 byte).

**Pipeline de Leitura (Geração):**
1. O LLM pede o KV da camada. Lemos os Índices do CROM Cache.
2. Reconstruímos os Centroides $\hat{K}_{unrot}$.
3. Rotacionamos de volta para as posições reais: $\hat{K} = RoPE_m(\hat{K}_{unrot})$.
4. Entregamos ao FlashAttention.

### O Papel do PCA (Decorrelation)
Antes mesmo do VQ, se aplicarmos um PCA ortogonal ao cache (KVTC-style), garantimos que os eixos do espaço que têm maior variância (os Outliers Channels, conhecidos por destruírem LLM Quantization) sejam tratados em subespaços independentes.

---
**Critério de Sucesso do Lab16:** 
Demonstrar via script PyTorch que o erro de reconstrução do pipeline CommVQ é estritamente menor que o VQ cego ao usar matrizes simuladas do LLaMA.
