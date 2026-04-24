# PLANEJAMENTO — Pesquisa 4: Compressão 2D (Codebooks Quantizados)

> **Objetivo:** Alcançar o estado da arte em *Extreme Quantization* cruzando os paradigmas de Quantização Vetorial (Codebooks da Pesquisa 2/3) com Quantização Escalar (INT8/INT4), implementando um "Dicionário de Inteiros".
>
> **Data Prevista:** 2026
> **Status:** 💡 Ideação / Backlog

---

## 🎯 O Conceito Base (A Tese de Pedro)
Atualmente, nosso motor `CromLinear` usa índices `uint16` (leves) que apontam para um dicionário (Codebook) com pesos em `FP16` (pesados, porém precisos). 
A ideia central da Pesquisa 4 é: **"O Codebook não concorre com a quantização, ele pode ser feito a partir de um modelo quantizado."** 
Isso significa aplicar uma compressão de dupla dimensão:
1. **Dimensão Vetorial:** O Tensor inteiro vira apenas ponteiros numéricos (como K-Means).
2. **Dimensão Escalar:** O Codebook em si deixa de armazenar valores de ponto flutuante `FP16` e passa a armazenar inteiros `INT4` ou `INT8`.

---

## 📋 Backlog de Ideação (Checklist Futura)

### Eixo 1: K-Means em Espaço Quantizado
- [ ] 1. Estudar se rodamos o K-Means em pesos já reduzidos a `INT8`, ou se rodamos o K-Means em `FP16` e depois quantizamos os Centroides resultantes (Codebook) usando o método Min-Max para extrair `Zero Point` e `Scale`.
- [ ] 2. Implementar função `quantize_codebook(codebook_fp16) -> (codebook_int8, scale, zero_point)`.

### Eixo 2: Reconstrução Dupla no Forward (Inference)
- [ ] 1. Alterar a lógica matemática do `forward`. Em vez de fazer `W = codebook[indices]`, teremos que fazer a desquantização escalar on-the-fly:
  `W_fp16 = (codebook_int8[indices] - zero_point) * scale`
- [ ] 2. Avaliar o overhead de CPU. A soma dessa desquantização pode ser resolvida usando AVX512 (ou OneBLAS se mantido Pythonico).

### Eixo 3: QAT (Quantization Aware Training) 2D
- [ ] 1. Retomar a nuvem (Pesquisa 2) para treinar um modelo nativamente em 2D. 
- [ ] 2. Adaptar o *Straight-Through Estimator* (STE) para lidar com o duplo arredondamento numérico (gradientes passando por dois gargalos de quantização simultâneos).

---

## 📊 Benefícios Esperados
- O Codebook em si encolherá em 2 a 4 vezes (de FP16 para INT8 ou INT4).
- Se o Llama-3 de 8B precisa de ~4GB hoje em PTQ Híbrido (Pesquisa 3), com a compressão de Pesquisa 4, pode ser viável operá-lo com ~2.5GB de RAM, cruzando a barreira dos hardwares limitadíssimos (como Raspberry Pi ou celulares velhos).

---
*Nota: Este planejamento ficará congelado até a conclusão empírica e redação do `papel0.md` e dos benchmarks da Pesquisa 3.*
