#!/bin/bash
# 🚀 CROM Neural Cockpit v3.0 - The Dynamic Flight Interface

# 1. Configurações de Path (Blindadas)
WORKSPACE="/home/j/Área de trabalho/crompressor-neuronio/pesquisas/testes"
MNT_DIR="../fuse_mnt"
LLAMA_CLI="/home/j/Área de trabalho/crompressor-ia/pesquisa/poc_llama_cpp_fuse/llama.cpp/build/bin/llama-cli"
GGUF="${MNT_DIR}/virtual_brain.gguf"
INF_LOG="${WORKSPACE}/inference.tmp"

cd "$WORKSPACE" || exit 1

# 2. Utilitários de Limpeza
function kill_zombies() {
    killall fuse_demon_check 2>/dev/null
    killall llama-cli 2>/dev/null
    fusermount3 -uz "$MNT_DIR" 2>/dev/null || true
    rm -f "$INF_LOG"
}

function exit_handler() {
    tput cnorm # Restaura o cursor
    echo -e "\n\e[31m[!] Ejetando do Cockpit...\e[0m"
    kill_zombies
    exit 0
}

trap exit_handler SIGINT SIGTERM

# 3. Preparação
trap exit_handler EXIT
kill_zombies
tput civis # Esconde o cursor

echo -e "\e[33m[+] Compilando Engine...\e[0m"
go build -o fuse_demon_check ./cmd/fuse_test/main.go
./fuse_demon_check > fuse_daemon.log 2>&1 &
FUSE_PID=$!
sleep 4

# 4. Renderizadores
function draw_heatmap() {
    local stats=$(curl -s "http://localhost:9999/stats")
    tput cup 4 0
    echo -e "\e[36m╔═════════════════════════════════════════════════════════════╗"
    echo -e "║   🧠 DELTA HEATMAP - TOPOLOGIA CEREBRAL (REAL-TIME)         ║"
    echo -e "╚═════════════════════════════════════════════════════════════╝\e[0m"
    
    if [[ -z "$stats" || "$stats" == "{}" ]]; then
        echo -e "   [💤] Cérebro em repouso. Aguardando atividade...\n"
    else
        # Renderização ultrarrápida via JQ Pipeline
        echo -n "   "
        echo "$stats" | jq -j '
            range(0; 448) as $i | 
            (.[$i|tostring] // 0) as $c |
            (if $c == 0 then "\u001b[90m.\u001b[0m"
            elif $c < 5 then "\u001b[34m+\u001b[0m"
            elif $c < 15 then "\u001b[33m#\u001b[0m"
            else "\u001b[31m@\u001b[0m" end) +
            (if (($i + 1) % 64 == 0) then "\n   " else "" end)
        '
        echo ""
    fi
}

function draw_inference_pane() {
    tput cup 15 0
    echo -e "\e[35m╔═════════════════════════════════════════════════════════════╗\e[0m"
    echo -e "\e[35m║   💬 MODEL OUTPUT (LIVE FEED)                               ║\e[0m"
    echo -e "\e[35m╚═════════════════════════════════════════════════════════════╝\e[0m"
    if [ -f "$INF_LOG" ]; then
        # Mostra as últimas 8 linhas do output, filtrando lixo de escape
        tail -n 8 "$INF_LOG" | sed 's/\x1b\[[0-9;]*m//g' | cut -c 1-80
    else
        echo "   [⏳] Nenhuma inferência ativa. Pressione [space] para bipar."
    fi
}

function draw_controls() {
    tput cup 27 0
    echo "---------------------------------------------------------------"
    echo -e " \e[1mPersona Ativa:\e[0m \e[32m$CURRENT_PERSONA\e[0m"
    echo -e " \e[1m[1-4]\e[0m Mudar | \e[1m[S]\e[0m Novo Prompt | \e[1m[Q]\e[0m Sair"
    echo "---------------------------------------------------------------"
}

# 5. Loop Principal do Cockpit
clear
CURRENT_PERSONA="BASE"

while true; do
    draw_heatmap
    draw_inference_pane
    draw_controls
    
    # Captura Input (Non-blocking)
    read -t 0.5 -n 1 key
    case $key in
        1) curl -s "http://localhost:9999/context?persona=base" >/dev/null; CURRENT_PERSONA="BASE" ;;
        2) curl -s "http://localhost:9999/context?persona=code" >/dev/null; CURRENT_PERSONA="CODE" ;;
        3) curl -s "http://localhost:9999/context?persona=math" >/dev/null; CURRENT_PERSONA="MATH" ;;
        4) curl -s "http://localhost:9999/context?persona=creative" >/dev/null; CURRENT_PERSONA="CREATIVE" ;;
        s|S)
            tput cnorm; tput cup 30 0; 
            read -p "💬 Novo Prompt: " prompt
            read -p "⏳ Token Limit (default 256): " tlimit
            tlimit=${tlimit:-256}
            rm -f "$INF_LOG"
            # Dispara em background
            "$LLAMA_CLI" -m "$GGUF" -p "$prompt" -n "$tlimit" --no-mmap > "$INF_LOG" 2>&1 &
            INF_PID=$!
            tput civis
            ;;
        q|Q) exit_handler ;;
    esac

    # Verifica se o processo de fundo morreu para limpar o log se não quisermos mais ver
    if [[ -n "$INF_PID" ]]; then
         if ! ps -p $INF_PID > /dev/null; then
             unset INF_PID
             echo -e "\n\e[32m[✔] Inferência concluída.\e[0m" >> "$INF_LOG"
         fi
    fi
done
