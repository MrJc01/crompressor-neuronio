#!/bin/bash

VAST_PORT="16410"
VAST_IP="ssh4.vast.ai"
INSTANCE_ID="35526411"
SSH_KEY="~/.ssh/id_rsa"

echo "🔍 Iniciando radar! Buscando resultados a cada 10 segundos..."

while true; do
  # 1. Verifica se o arquivo de output começou a ser criado
  if ssh -o StrictHostKeyChecking=no -p $VAST_PORT -i $SSH_KEY root@$VAST_IP "ls /workspace/phi3_crom.cromv3" >/dev/null 2>&1; then
     
     # 2. Se o arquivo existe, precisamos saber se o Python já terminou de escrever nele
     if ssh -o StrictHostKeyChecking=no -p $VAST_PORT -i $SSH_KEY root@$VAST_IP "pgrep -f ptq_compressor.py" >/dev/null 2>&1; then
        echo "⏳ [$(date +'%H:%M:%S')] Arquivo gerado, mas a compressão/gravação ainda está em andamento. Aguardando..."
     else
        echo "✅ [$(date +'%H:%M:%S')] COMPRESSÃO FINALIZADA COM SUCESSO!"
        
        echo "⬇️ Baixando o modelo comprimido para o seu PC..."
        scp -P $VAST_PORT -i $SSH_KEY root@$VAST_IP:/workspace/phi3_crom.cromv3 ./phi3_crom.cromv3
        
        echo "💥 Destruindo a máquina na Vast.ai para proteger seu saldo de \$4..."
        vastai destroy instance $INSTANCE_ID
        
        echo "🎉 TUDO PRONTO! Agora você pode rodar localmente:"
        echo "python local_chat.py phi3_crom.cromv3"
        break
     fi
  else
     echo "⏳ [$(date +'%H:%M:%S')] Instalando dependências ou processando tensores na nuvem... Aguardando."
  fi
  
  sleep 10
done
