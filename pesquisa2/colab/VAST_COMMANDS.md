# Vast.ai — Comandos Copy-Paste

## 1. Alugar Máquina
- Template: **PyTorch** (qualquer)
- GPU: **RTX 3090** (~$0.15-0.25/hr)
- Custo total estimado: ~$0.50-0.75 (2-3h)

## 2. Colar no Terminal (1 bloco só)

```bash
cd /workspace && git clone https://github.com/MrJc01/crompressor-neuronio && cd crompressor-neuronio/pesquisa2 && pip install -q transformers datasets && python colab/vast_optimized.py 2>&1 | tee vast_log.txt
```

## 3. Quando Terminar — Baixar Resultados

```bash
# No terminal do Vast.ai:
cat resultados/vast_results.json

# Ou copie os arquivos:
# - checkpoints_vast/cromgpt.cromv3
# - checkpoints_vast/baseline.pt  
# - resultados/vast_results.json
# - vast_log.txt
```

## 4. Alternativa: Baixar via SCP
```bash
# No seu PC local:
scp -P PORTA root@IP_VAST:/workspace/crompressor-neuronio/pesquisa2/resultados/vast_results.json .
scp -P PORTA root@IP_VAST:/workspace/crompressor-neuronio/pesquisa2/vast_log.txt .
```

## Tempo Estimado
- Download dados: ~5min
- Treino CromGPT (8K steps): ~60-90min  
- Treino Baseline (8K steps): ~60-90min
- Avaliação + geração: ~5min
- **Total: ~2-3h**

## Se Der Erro de Memória
Edite MAX_STEPS e BATCH_SIZE no vast_optimized.py antes de rodar.
