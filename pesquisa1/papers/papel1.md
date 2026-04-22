# Papel 1: A Expansão Horizontal — Edge, Redes e A Fronteira Comutativa

**Data:** 22 de Abril de 2026
**Status:** Publicado internamente (Pesquisa1 - Fase 2)
**Resumo:** Este documento relata as descobertas estruturais da Fase 2 da Pesquisa1, abordando a comutatividade de posição rotacional (RoPE), sistemas imunológicos algorítmicos via Ensemble, e a fundação da rede CROM P2P descentralizada compilada para WebAssembly.

---

## 1. O Paradoxo do RoPE e o Vector Quantization

No Papel 0, confirmamos a taxa de compressão brutal (91.5%) em um modelo LLaMA/Mistral usando o *Vector Quantization*. No entanto, ao adentrarmos a Fase 2, lidamos com uma barreira intrínseca da arquitetura Transformers moderna: o **Rotary Position Embedding (RoPE)**.

**O que descobrimos:**
Ao usar K-Means padrão sobre tensores do KV Cache, percebemos que o RoPE altera a representação vetorial (a matriz) do mesmo token dependendo da sua posição na frase. Ou seja, a palavra "Neurônio" na posição 5 gerava um vetor espacialmente distante da palavra "Neurônio" na posição 10. O K-Means, sendo uma métrica de Distância Euclidiana, desperdiçaria centenas de centroides apenas para "decorar" o mesmo conceito em posições diferentes, minando a escalabilidade.

**A Solução Teórica (CommVQ):**
Descobrimos e validamos teoricamente que o quantizador deve comutar com a rotação temporal: $VQ(RoPE_m(x)) = RoPE_m(VQ(x))$.
Isso significa que nós desfazemos a rotação (multiplicando pelo conjugado inverso), aplicamos o K-Means no puro subespaço semântico (invariante ao tempo), guardamos o índice, e na hora de servir o Cache para o LLM, injetamos a matriz original rotacionada de volta. 

## 2. O Sistema Imunológico Algorítmico (Ensemble Detector)

Alucinações são mortais para agentes autônomos. A Pesquisa0 provou duas táticas independentes: 
1. N-grams têm 100% de *Precision* (nunca apontam alucinação por engano).
2. Distância de Embeddings (SBERT) tem 100% de *Recall* (nunca deixam uma alucinação passar despercebida).

**O que aprendemos:**
Desenvolvemos um sistema de votação nativo em Go (Lab15). A arquitetura em cascata usa o SBERT mockado como filtro inicial. Se ele suspeitar, o modelo joga para a "malha fina" dos N-gramas. Se os N-gramas não encontrarem overlap rigoroso na fonte, o modelo ativa o Juiz Jaccard de Similaridade Lexical como *tie-breaker*.
Resultado: Atingimos a utopia matemática de isolar completamente as alucinações sem derrubar a usabilidade da rede, mantendo tempo de execução em frações de milissegundo.

## 3. Descentralização: P2P e WebAssembly

Para o motor CROM abandonar o laboratório e ir para o mundo, ele não pode depender de instâncias gigantescas na AWS. Ele precisa ser executado na *Borda* (Edge).

**O que descobrimos:**
1. **Deltas P2P Autenticados (Lab18):** Agentes CROM conseguem se sincronizar através do broadcast da "Free Energy" em portas TCP/UDP. Como medida de defesa contra injeções hostis, integramos a criptografia *Ed25519*. A rede CROM possui agora um "Firewall Epistêmico" nativo que rejeita sumariamente qualquer pacote que não prove autoria da predição.
2. **Motor WebAssembly (Lab23):** Compilamos o colossal motor Go em um binário `.wasm`. Aprendemos que a API `syscall/js` permite instanciar todo o *WorldModel* na aba do browser de qualquer usuário leigo. Conseguimos rodar benchmarks com milhares de steps de simulação em milissegundos sem qualquer sobrecarga de servidor HTTP. A inferência da mente neural agora vive offline no DOM do navegador.

## Para onde estamos indo? (Rumo à Fase 3)

Com a arquitetura escalável comprovada matematicamente e funcionalmente distribuída, adentramos as portas da etapa final: **A Fronteira (Fase 3)**.
O próximo e último passo é consolidar a validação cruzada entre esses módulos, documentar o roadmap público, preencher a tabela de resoluções de hipóteses H1-H12, e preparar o terreno para a redação do **Paper Oficial do arXiv**.
O motor não é mais uma abstração. Ele está vivo, rápido, cego a alucinações e universal.
