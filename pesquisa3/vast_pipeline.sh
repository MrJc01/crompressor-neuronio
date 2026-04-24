#!/bin/bash

# ==============================================================================
# Pipeline de Compressão PTQ Híbrido - Vast.ai
# ==============================================================================
# Este script automatiza o envio, execução e resgate do modelo comprimido
# em uma instância remota da Vast.ai.
# ==============================================================================

# Defina o IP e a Porta da sua instância Vast.ai aqui:
VAST_IP="adicione_o_ip_aqui"
VAST_PORT="adicione_a_porta_aqui"
SSH_KEY="~/.ssh/id_rsa"

MODEL_ID="microsoft/Phi-3-mini-4k-instruct"
OUTPUT_FILE="phi3_crom.cromv3"

echo "🚀 Iniciando Pipeline Vast.ai para o modelo: $MODEL_ID"

# 1. Enviar o script de compressão para a nuvem
echo "📦 1. Fazendo upload do ptq_compressor.py para a nuvem..."
scp -P $VAST_PORT -i $SSH_KEY ./ptq_compressor.py root@$VAST_IP:/workspace/

# 2. Instalar dependências e executar a compressão via SSH
echo "⚙️ 2. Executando o compressor na nuvem (Isso pode levar alguns minutos)..."
ssh -p $VAST_PORT -i $SSH_KEY root@$VAST_IP << 'EOF'
    # Entra no diretório de trabalho
    cd /workspace/
    
    # Instala o FAISS para K-Means ultrarrápido
    pip install faiss-gpu faiss-cpu transformers torch accelerate numpy
    
    # Executa o compressor PTQ Híbrido
    # Parâmetros recomendados para 8GB RAM local: K=256, D=8
    python ptq_compressor.py --model "microsoft/Phi-3-mini-4k-instruct" --out "phi3_crom.cromv3" --k 256 --d 8
EOF

# 3. Baixar o artefato .cromv3 gerado
echo "⬇️ 3. Fazendo o download do modelo comprimido (.cromv3)..."
scp -P $VAST_PORT -i $SSH_KEY root@$VAST_IP:/workspace/$OUTPUT_FILE ./$OUTPUT_FILE

echo "🎉 Pipeline Concluído! O arquivo $OUTPUT_FILE está pronto para uso local."
echo "⚠️ Lembre-se de destruir sua instância na Vast.ai para evitar cobranças."
