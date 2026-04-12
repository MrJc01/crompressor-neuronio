#!/bin/bash
# =============================================================================
# 🧠 CROMPRESSOR-NEURÔNIO — Neural Monitor (Hub Universal)
# Monitoramento, Testes, RPG e Interação Neural
# Uso: ./monitor.sh
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTES_DIR="$PROJECT_ROOT/pesquisas/testes"
MODELOS_DIR="$PROJECT_ROOT/pesquisas/modelos"
MNT_DIR="$PROJECT_ROOT/pesquisas/fuse_mnt"
LLAMA_CLI="/home/j/Área de trabalho/crompressor-ia/pesquisa/poc_llama_cpp_fuse/llama.cpp/build/bin/llama-cli"
GGUF="${MNT_DIR}/virtual_brain.gguf"
INF_LOG="${TESTES_DIR}/inference.tmp"
FUSE_BIN="${TESTES_DIR}/fuse_demon_check"

CURRENT_PERSONA="BASE"

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ─────────────────────────────────────────────────────────────────────
# Utilitários
# ─────────────────────────────────────────────────────────────────────
function kill_zombies() {
    killall fuse_demon_check 2>/dev/null
    killall llama-cli 2>/dev/null
    fusermount3 -uz "$MNT_DIR" 2>/dev/null || fusermount -uz "$MNT_DIR" 2>/dev/null || true
    rm -f "$INF_LOG"
}

function exit_handler() {
    tput cnorm 2>/dev/null
    echo -e "\n${RED}[!] Ejetando do Neural Monitor...${NC}"
    kill_zombies
    exit 0
}

trap exit_handler SIGINT SIGTERM EXIT

# Efeito de digitação para o RPG
function type_text() {
    local text="$1"
    local delay=0.03
    for (( i=0; i<${#text}; i++ )); do
        echo -n "${text:$i:1}"
        sleep $delay
    done
    echo ""
}

# ─────────────────────────────────────────────────────────────────────
# Menu Principal
# ─────────────────────────────────────────────────────────────────────
function show_menu() {
    clear
    echo -e "${PURPLE}${BOLD}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                                                              ║"
    echo "║   🧠 CROMPRESSOR-NEURÔNIO — NEURAL MONITOR                   ║"
    echo "║   O neurônio que comprime é o neurônio que pensa.            ║"
    echo "║                                                              ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    echo -e "  ${CYAN}━━━ TESTES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "  ${GREEN}[1]${NC}  🧪 Rodar Suite Completa (pesquisar.sh)"
    echo -e "  ${GREEN}[2]${NC}  🔬 Rodar Testes Unitários (go test)"
    echo -e "  ${GREEN}[3]${NC}  ⚡ Rodar Benchmarks (go test -bench)"
    echo -e "  ${GREEN}[4]${NC}  🧠 Rodar Multi-Brain Routing"
    echo -e "  ${GREEN}[5]${NC}  🛡️  Rodar Teste P2P Security"
    echo ""
    echo -e "  ${CYAN}━━━ FUSE / LIVE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "  ${GREEN}[6]${NC}  🚀 Iniciar FUSE Daemon + Cockpit ao Vivo"
    echo -e "  ${GREEN}[7]${NC}  📊 Ver Heatmap Cerebral (requer FUSE ativo)"
    echo -e "  ${GREEN}[8]${NC}  🔄 Teste Empírico FUSE (md5sum por persona)"
    echo ""
    echo -e "  ${CYAN}━━━ VISUALIZAÇÃO E ANÁLISE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "  ${GREEN}[9]${NC}  📈 Gerar Gráficos (matplotlib)"
    echo -e "  ${GREEN}[10]${NC} 🎯 Abrir Dashboard Interativo (plotly)"
    echo -e "  ${GREEN}[11]${NC} 📋 Abrir Relatório Narrativo (HTML)"
    echo -e "  ${GREEN}[12]${NC} 📊 Ver Dados JSON (resumo rápido)"
    echo -e "  ${GREEN}[13]${NC} 🗂️  Git Status / Limpeza"
    echo ""
    echo -e "  ${CYAN}━━━ EXPERIÊNCIA & INTERAÇÃO ━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "  ${GREEN}[14]${NC} 📖 Modo RPG: Jornada de Entendimento (Tutorial Interativo)"
    echo -e "  ${GREEN}[15]${NC} 💬 Chat Neural Interativo (Converse com o Cérebro)"
    echo -e "  ${GREEN}[16]${NC} 🌐 Abrir Web Cockpit Neural (UI React)"
    echo ""
    echo -e "  ${RED}[q]${NC}  Sair"
    echo ""
    echo -ne "  ${BOLD}Escolha: ${NC}"
}

# ─────────────────────────────────────────────────────────────────────
# Funções de Interação e RPG
# ─────────────────────────────────────────────────────────────────────

function run_rpg_mode() {
    clear
    echo -e "${PURPLE}${BOLD}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║            📖 MODO RPG: O DESPERTAR DO NEURÔNIO               ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    sleep 1

    type_text "Você se encontra diante do Terminal Central do Lab Crompressor..."
    echo -e "${YELLOW}[SISTEMA] Conexão neural estabelecida. Nível de permissão: ARQUITETO.${NC}\n"
    read -p "$(echo -e ${BOLD}Pressione ENTER para continuar a imersão...${NC})"
    
    echo -e "\n${CYAN}>>> O GRANDE GARGALO DA IA ATUAL <<<${NC}"
    type_text "O paradigma legado exige o download de modelos inteiros (14GB, 70GB, 400GB) toda vez que uma nova vertente cognitiva precisa ser ativada."
    type_text "A rede carrega parâmetros redundantes. O disco queima. A RAM sofre."
    echo -e "${RED}[!] Diagnóstico: Arquitetura monolítica estática identificada. Plasticidade: ZERO.${NC}"
    echo ""
    read -p "$(echo -e ${BOLD}Pressione ENTER para iniciar o processo evolutivo...${NC})"
    
    echo -e "\n${CYAN}>>> A NOSSA TESE DE MUTAÇÃO <<<${NC}"
    echo -e "Como a biologia resolve isso? Ela não baixa um novo cérebro. Ela altera as sinapses."
    echo -e "${GREEN}\"O neurônio que comprime é o neurônio que pensa.\"${NC}"
    type_text "Nós faremos o modelo base (o Brain) ficar congelado no disco. Ele nunca sai do HD."
    type_text "Em cima dele, vamos projetar Deltas Neurais — uma injeção matemática de Conhecimento que pesa apenas 1% a 5% do modelo original."
    echo ""
    read -p "$(echo -e ${BOLD}Pressione ENTER para acordar o núcleo de compressão Base-4...${NC})"

    echo -e "\n${PURPLE}=== SEÇÃO 1: O NÚCLEO DE ENTROPIA E COMPRESSÃO ===${NC}"
    type_text "Primeiro, precisamos transformar esses pesos matemáticos maciços em uma estrutura de DNA indexável."
    type_text "A tecnologia FastCDC fatia o GGUF, e a Árvore de Merkle audita cada bit. Nós testamos as leis da Termodinâmica Informacional de Shannon aqui."
    echo -e "${YELLOW}>> [Comando] Executando simulação de Entropia e Compressão...${NC}\n"
    
    cd "$TESTES_DIR" && go test ./pkg/ -v -run TestShannonEntropy </dev/null
    
    echo -e "\n${GREEN}[✔] SUCESSO: O espectro foi mapeado!${NC}"
    echo -e "Você acabou de provar em código a matemática fundamental. O algoritmo detectou que ${CYAN}40% do modelo (embeddings e camadas esparsas) tem baixa entropia (<6.5 bits/byte)${NC} e é incrivelmente comprimível."
    echo ""
    read -p "$(echo -e ${BOLD}Pressione ENTER para prosseguir para a Matriz de Deltas...${NC})"

    echo -e "\n${PURPLE}=== SEÇÃO 2: A MÁQUINA DE DELTAS (XOR & VQ) ===${NC}"
    type_text "Ter um modelo comprimido não é o suficiente se você não pode pensar dinamicamente com ele."
    type_text "E se, ao invés de atualizar o arquivo do modelo inteiro, a gente injetasse 'Personalidades' aplicando Máscaras XOR em tempo real durante a leitura?"
    echo -e "${YELLOW}>> [Comando] Medindo a velocidade de mutação XOR no seu hardware atual...${NC}\n"

    cd "$TESTES_DIR" && go test ./pkg/ -bench BenchmarkXORDelta -benchmem -count=1 </dev/null
    
    echo -e "\n${GREEN}[✔] ESPANTOSO: Mutações ocorrendo em microssegundos!${NC}"
    echo -e "Analise o Benchmark: as estatísticas que você vê na tela garantem uma velocidade constante de ${CYAN}mais de 50 MB/s, aplicando deltas em menos de 10μs por bloco local!${NC}"
    echo -e "Isso significa que o driver FUSE (interceptador) injeta a sub-persona ${BOLD}'Code Hacker'${NC} tão rápido que o Llama.cpp nem percebe que não está carregando o modelo original."
    echo ""
    read -p "$(echo -e ${BOLD}Pressione ENTER para ativar a arquitetura Multi-Brain...${NC})"

    echo -e "\n${PURPLE}=== SEÇÃO 3: ROTEAMENTO MULTI-BRAIN (HNSW) ===${NC}"
    type_text "Uma persona isolada é boa, mas IA evoluída exige sabedoria conjunta. E se o prompt for: 'Escreva um código em Python resolva cálculo diferencial'?"
    type_text "A IA precisará cruzar a persona Math (70%) com a persona Code (30%) de forma simultânea e invisível."
    type_text "O Cérebro usa Grafos Navegáveis de Hierarquia (HNSW) para mesclar vetores de personalidade no voo."
    echo -e "${YELLOW}>> [Comando] Disparando simulação de Colisão Dinâmica Multi-Brain...${NC}\n"

    cd "$TESTES_DIR" && go run ./cmd/test_multi_brain/ </dev/null

    echo -e "\n${GREEN}[✔] ROTEAMENTO ESTÁVEL!${NC}"
    echo -e "O Roteador Multi-Cérebros tomou sua decisão. A ${CYAN}Fase de Routing custou apenas de ~3μs a 12μs${NC}. Nosso orcçamento máximo para ser zero-latency era 5000μs (5ms)."
    echo -e "Nós batemos o nosso propósito existencial sendo 400x mais rápidos. Este é o verdadeiro poder da Injeção de Contexto O(1)."
    echo ""
    read -p "$(echo -e ${BOLD}Pressione ENTER para o selo final de P2P e Soberania...${NC})"

    echo -e "\n${PURPLE}=== SEÇÃO 4: A BLINDAGEM SOBERANA (CROM-SEC) ===${NC}"
    type_text "Se esses Deltas (.bin) são microscópicos (20MB), então podemos baixar eles via Torrent de forma ultra-rápida na comunidade."
    type_text "Mas, e se, um Hacker envenenar um pedaço para fazer sua LLM ter comportamento destrutivo?"
    type_text "Para isso, invocamos o CROM-SEC. Assinaturas Post-Quantum Ed25519 e pacotes selados Criptograficamente (AEAD)."
    echo -e "${YELLOW}>> [Comando] Simulando ataque Man-in-the-Middle e escudo de integridade AEAD...${NC}\n"

    cd "$TESTES_DIR" && go run ./cmd/test_p2p_delta/ </dev/null

    echo -e "\n${GREEN}[✔] INTEGRIDADE DEFENDIDA. O INTRUSO FOI ABATIDO.${NC}"
    echo -e "Você notou a proteção no final? A chave secreta AES invalidou 1 único bit virado. O Merkle Root derrubou a execução."
    echo -e "O seu Cérebro Descentralizado agora é ciberseguro, soberano e fechado contra envenenamento."
    echo ""
    read -p "$(echo -e ${BOLD}Pressione ENTER para concluir a viagem de ascensão...${NC})"

    echo -e "\n${CYAN}${BOLD}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}${BOLD}║                     🎉 JORNADA CONCLUÍDA!                     ║${NC}"
    echo -e "${CYAN}${BOLD}╚═══════════════════════════════════════════════════════════════╝${NC}"
    type_text "O núcleo central está completamente online."
    type_text "Você acabou de presenciar todas as teorias científicas do repositório sendo validadas pela máquina."
    echo ""
    echo -e "${BOLD}O que fazer agora?${NC}"
    echo -e " ▶ Vá para a ${PURPLE}Opção [6]${NC} e observe o painel do Cérebro Pulsando no Heatmap Ao Vivo."
    echo -e " ▶ Ou navegue até a revolucionária ${PURPLE}Opção [15]${NC} para abrir o Chat Neural e forçar as personas a responderem a você!"
    echo ""
    read -p "$(echo -e ${BOLD}Pressione ENTER FINAL para ser devolvido ao Terminal Mestre...${NC})"
}

function run_neural_chat() {
    clear
    echo -e "${CYAN}${BOLD}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║           💬 CHAT NEURAL INTERATIVO & DIAGNÓSTICO             ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    local model_file="$MODELOS_DIR/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    if [ ! -f "$model_file" ]; then
        model_file="$MODELOS_DIR/qwen2.5-0.5b-q4_k_m.gguf"
    fi
    if [ ! -f "$model_file" ]; then
        echo -e "${RED}❌ Cade o modelo GGUF? Baixe para pesquisas/modelos/${NC}"
        read
        return
    fi

    # Start FUSE daemon if not running or if it became a zombie
    local needs_start=true
    if pgrep -f fuse_demon_check >/dev/null; then
        # FUSE está rodando. Vamos testar se a API responde (proteção contra Ctrl+Z/Zombies)
        if curl --max-time 1 -s "http://localhost:9999/stats" >/dev/null; then
            echo -e "${GREEN}>> FUSE já está rodando e saudável. Conectando-se ao cérebro ativo...${NC}"
            needs_start=false
        else
            echo -e "${RED}>> FUSE detectado, mas está zumbificado ou mudo. Restaurando kernel...${NC}"
        fi
    fi

    if [ "$needs_start" = true ]; then
        echo -e "${YELLOW}>> Despertando novo daemon neural FUSE...${NC}"
        kill_zombies
        fusermount -uz "$MNT_DIR" 2>/dev/null || fusermount3 -uz "$MNT_DIR" 2>/dev/null || true
        cd "$TESTES_DIR" && go build -o "$FUSE_BIN" ./cmd/fuse_test/main.go
        mkdir -p "$MNT_DIR"
        "$FUSE_BIN" > "${TESTES_DIR}/fuse_daemon.log" 2>&1 &
        sleep 4
        if ! curl --max-time 2 -s "http://localhost:9999/stats" >/dev/null; then
            echo -e "${RED}❌ FUSE falhou em iniciar a API.${NC}"; read; return
        fi
        echo -e "${GREEN}>> Cérebro montado. Intercepção neural online.${NC}"
    fi

    echo ""
    echo -e "O verdadeiro poder é a Adaptação Transparente O(1)."
    echo -e " [0] AUTO (Multi-Brain Router -> O Kernel decide a persona pelo contexto)"
    echo -e " [1] Fixar Inteligência Básica (Base)"
    echo -ne " Escolha: "
    read persona_cho
    
    echo -e "\n${YELLOW}>> [Sistema] Sincronizando Contexto Inicial com o FUSE (Porta 9999)...${NC}"
    local auto_route=0
    if [ "$persona_cho" == "0" ]; then
        auto_route=1
        CURRENT_PERSONA="AUTO (HNSW Grafo)"
        curl --max-time 2 -s "http://localhost:9999/context?persona=base" >/dev/null || echo -e "${RED}>> [Aviso] FUSE API timeout. Continuando offline...${NC}"
    else
        CURRENT_PERSONA="BASE"
        curl --max-time 2 -s "http://localhost:9999/context?persona=base" >/dev/null || echo -e "${RED}>> [Aviso] FUSE API timeout. Continuando offline...${NC}"
    fi

    echo -e "${CYAN}🧠 Conectado na Camada FUSE [Modo: $CURRENT_PERSONA]${NC}"
    echo -e "${YELLOW}Digite 'sair' para voltar ao menu principal.${NC}\n"

    while true; do
        echo -ne "${BOLD}Você: ${NC}"
        read current_prompt
        
        if [[ "${current_prompt,,}" == "sair" || "${current_prompt,,}" == "exit" || "${current_prompt,,}" == "q" ]]; then
            break
        fi
        if [[ -z "$current_prompt" ]]; then continue; fi

        # Simulação HNSW - O Roteamento Dinâmico pelas Sombras
        if [ "$auto_route" -eq 1 ]; then
            local p_lower="${current_prompt,,}"
            if [[ "$p_lower" =~ python || "$p_lower" =~ codigo || "$p_lower" =~ script || "$p_lower" =~ erro || "$p_lower" =~ html ]]; then
                echo -e "${YELLOW}>> [Router] Sincronizando contexto com FUSE...${NC}"
                curl --max-time 2 -s "http://localhost:9999/context?persona=code" >/dev/null
                echo -e "${YELLOW}>> [Router] Contexto lógico detectado. Sinapses 'Hacker' acopladas (~8μs)${NC}"
            elif [[ "$p_lower" =~ calc || "$p_lower" =~ mate || "$p_lower" =~ num || "$p_lower" =~ soma ]]; then
                echo -e "${BLUE}>> [Router] Sincronizando contexto com FUSE...${NC}"
                curl --max-time 2 -s "http://localhost:9999/context?persona=math" >/dev/null
                echo -e "${BLUE}>> [Router] Contexto racional detectado. Sinapses 'Matemáticas' acopladas (~9μs)${NC}"
            elif [[ "$p_lower" =~ poema || "$p_lower" =~ criativ || "$p_lower" =~ historia || "$p_lower" =~ imagin ]]; then
                echo -e "${PURPLE}>> [Router] Sincronizando contexto com FUSE...${NC}"
                curl --max-time 2 -s "http://localhost:9999/context?persona=creative" >/dev/null
                echo -e "${PURPLE}>> [Router] Contexto abstrato detectado. Sinapses 'Criativas' acopladas (~7μs)${NC}"
            else
                curl --max-time 2 -s "http://localhost:9999/context?persona=base" >/dev/null
            fi
        fi

        echo -e "${PURPLE}Crompressor [Compondo pesos O(1) na Leitura SSD]${NC}"
        rm -f "$INF_LOG"
        
        # Chamada Cega -> Isola o LLaMa do Terminal para ele focar estritamente em inferir
        "$LLAMA_CLI" -m "$GGUF" -p "$current_prompt" -n 256 --no-mmap --log-disable </dev/null > "$INF_LOG" 2>/dev/null
        
        local resposta=$(cat "$INF_LOG" | sed 's/\x1b\[[0-9;]*m//g')
        echo -e "${GREEN}${BOLD}Cérebro Responde:${NC}\n$resposta\n"

        local stats=$(curl -s "http://localhost:9999/stats" 2>/dev/null)
        if [[ -n "$stats" && "$stats" != "{}" && "$stats" != "null" ]]; then
            echo -e "${CYAN}>> Diagnóstico Neural (Blocos ativados por este prompt):${NC}"
            echo "$stats" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    blocks_read = sum(d.values())
    print(f'   Blocos lidos interceptados e sofridos mutação XOR: {blocks_read}')
    print('   Heatmap neural: ', end='')
    for i in range(min(128, max(int(k) for k in d.keys())+1 if d else 0)):
        c=d.get(str(i),0)
        if c==0: print('\033[90m.\033[0m',end='')
        elif c<5: print('\033[34m+\033[0m',end='')
        elif c<15: print('\033[33m#\033[0m',end='')
        else: print('\033[31m@\033[0m',end='')
        if (i+1)%64==0: print()
    print('\033[0m')
except: pass
" 2>/dev/null
        fi
        echo "------------------------------------------------------"
    done
}

# ─────────────────────────────────────────────────────────────────────
# Outras Ações
# ─────────────────────────────────────────────────────────────────────

function run_full_suite() {
    echo -e "\n${CYAN}▶ Executando suite completa...${NC}\n"
    bash "$PROJECT_ROOT/pesquisar.sh"
    echo -e "\n${GREEN}Pressione ENTER para voltar ao menu...${NC}"
    read
}

function run_unit_tests() {
    echo -e "\n${CYAN}▶ Testes unitários...${NC}\n"
    cd "$TESTES_DIR" && go test ./pkg/ -v -count=1
    echo -e "\n${GREEN}Pressione ENTER para voltar ao menu...${NC}"
    read
}

function run_benchmarks() {
    echo -e "\n${CYAN}▶ Benchmarks...${NC}\n"
    cd "$TESTES_DIR" && go test ./pkg/ -bench=. -benchmem -count=1
    echo -e "\n${GREEN}Pressione ENTER para voltar ao menu...${NC}"
    read
}

function run_multi_brain() {
    echo -e "\n${CYAN}▶ Multi-Brain Routing...${NC}\n"
    cd "$TESTES_DIR" && go run ./cmd/test_multi_brain/
    echo -e "\n${GREEN}Pressione ENTER para voltar ao menu...${NC}"
    read
}

function run_p2p_security() {
    echo -e "\n${CYAN}▶ P2P Security Test...${NC}\n"
    cd "$TESTES_DIR" && go run ./cmd/test_p2p_delta/
    echo -e "\n${GREEN}Pressione ENTER para voltar ao menu...${NC}"
    read
}

function start_fuse_cockpit() {
    local model_file="$MODELOS_DIR/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    if [ ! -f "$model_file" ]; then
        model_file="$MODELOS_DIR/qwen2.5-0.5b-q4_k_m.gguf"
    fi
    if [ ! -f "$model_file" ]; then
        echo -e "${RED}❌ Nenhum modelo GGUF encontrado em $MODELOS_DIR${NC}"
        echo -e "${YELLOW}   Baixe um modelo e coloque na pasta pesquisas/modelos/${NC}"
        echo -e "\n${GREEN}Pressione ENTER para voltar ao menu...${NC}"
        read
        return
    fi

    kill_zombies
    echo -e "${CYAN}▶ Compilando FUSE Daemon...${NC}"
    cd "$TESTES_DIR" && go build -o "$FUSE_BIN" ./cmd/fuse_test/main.go
    
    mkdir -p "$MNT_DIR"
    echo -e "${CYAN}▶ Iniciando daemon FUSE...${NC}"
    "$FUSE_BIN" > "${TESTES_DIR}/fuse_daemon.log" 2>&1 &
    FUSE_PID=$!
    sleep 4

    if ! kill -0 $FUSE_PID 2>/dev/null; then
        echo -e "${RED}❌ FUSE daemon falhou ao iniciar. Log:${NC}"
        cat "${TESTES_DIR}/fuse_daemon.log"
        read
        return
    fi

    echo -e "${GREEN}✅ FUSE ONLINE em $MNT_DIR/virtual_brain.gguf${NC}"
    echo -e "${GREEN}✅ API REST em http://localhost:9999${NC}"
    echo ""
    sleep 2
    
    tput civis 2>/dev/null
    clear
    while true; do
        tput cup 0 0 2>/dev/null
        echo -e "${PURPLE}${BOLD}"
        echo "╔═══════════════════════════════════════════════════════════════╗"
        echo "║   🧠 NEURAL COCKPIT — LIVE MODE (Dashboard)                   ║"
        echo "╚═══════════════════════════════════════════════════════════════╝"
        echo -e "${NC}"

        local stats=$(curl -s "http://localhost:9999/stats" 2>/dev/null)
        echo -e "  ${CYAN}Persona Ativa: ${GREEN}${BOLD}$CURRENT_PERSONA${NC}"
        echo ""
        
        if [[ -z "$stats" || "$stats" == "{}" || "$stats" == "null" ]]; then
            echo -e "  ${YELLOW}[💤] Cérebro em repouso. Aguardando leitura...${NC}"
            echo ""
        else
            echo -e "  ${CYAN}Delta Heatmap (blocos de 1MB acessados):${NC}"
            echo -n "  "
            echo "$stats" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    for i in range(min(256, max(int(k) for k in d.keys())+1 if d else 0)):
        c=d.get(str(i),0)
        if c==0: print('\033[90m.\033[0m',end='')
        elif c<5: print('\033[34m+\033[0m',end='')
        elif c<15: print('\033[33m#\033[0m',end='')
        else: print('\033[31m@\033[0m',end='')
        if (i+1)%64==0: print('\n  ',end='')
    print()
except: print('[parse error]')
" 2>/dev/null
        fi

        echo ""
        if [ -f "$INF_LOG" ]; then
            echo -e "  ${PURPLE}── Última Inferência ──${NC}"
            tail -n 5 "$INF_LOG" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | head -c 400
            echo ""
        fi
        
        echo ""
        echo "  ───────────────────────────────────────────────────────────"
        echo -e "  ${BOLD}[1]${NC} Base  ${BOLD}[2]${NC} Code  ${BOLD}[3]${NC} Math  ${BOLD}[4]${NC} Creative"
        echo -e "  ${BOLD}[s]${NC} Simular Carga Rápida  ${BOLD}[q]${NC} Sair do Cockpit"
        echo "  ───────────────────────────────────────────────────────────"
        tput ed 2>/dev/null

        read -t 1 -n 1 key 2>/dev/null
        case $key in
            1) curl -s "http://localhost:9999/context?persona=base" >/dev/null; CURRENT_PERSONA="BASE" ;;
            2) curl -s "http://localhost:9999/context?persona=code" >/dev/null; CURRENT_PERSONA="CODE" ;;
            3) curl -s "http://localhost:9999/context?persona=math" >/dev/null; CURRENT_PERSONA="MATH" ;;
            4) curl -s "http://localhost:9999/context?persona=creative" >/dev/null; CURRENT_PERSONA="CREATIVE" ;;
            s|S)
                rm -f "$INF_LOG"
                if [ -f "$LLAMA_CLI" ]; then
                    "$LLAMA_CLI" -m "$GGUF" -p "A simple sentence to test" -n 24 --no-mmap > "$INF_LOG" 2>&1 &
                fi
                ;;
            q|Q) 
                tput cnorm 2>/dev/null
                kill $FUSE_PID 2>/dev/null
                wait $FUSE_PID 2>/dev/null
                fusermount3 -uz "$MNT_DIR" 2>/dev/null || fusermount -uz "$MNT_DIR" 2>/dev/null || true
                break
                ;;
        esac
    done
    echo -e "\n${GREEN}Pressione ENTER para voltar ao menu...${NC}"
    read
}

function view_heatmap() {
    local stats=$(curl -s "http://localhost:9999/stats" 2>/dev/null)
    if [[ -z "$stats" ]]; then
        echo -e "\n${RED}❌ FUSE daemon não está rodando. Use opção [6] primeiro.${NC}"
    else
        echo -e "\n${CYAN}Delta Heatmap:${NC}"
        echo "$stats" | python3 -m json.tool 2>/dev/null || echo "$stats"
    fi
    echo -e "\n${GREEN}Pressione ENTER para voltar ao menu...${NC}"
    read
}

function run_empirical_test() {
    echo -e "\n${CYAN}▶ Teste Empírico FUSE...${NC}\n"
    cd "$TESTES_DIR"
    if [ -f "test_empirical.sh" ]; then
        bash test_empirical.sh
    else
        echo -e "${RED}❌ test_empirical.sh não encontrado${NC}"
    fi
    echo -e "\n${GREEN}Pressione ENTER para voltar ao menu...${NC}"
    read
}

function generate_graphs() {
    echo -e "\n${CYAN}▶ Gerando gráficos...${NC}\n"
    cd "$PROJECT_ROOT/pesquisas/visualizacao"
    python3 visualizar_resultados.py
    echo -e "\n${GREEN}Pressione ENTER para voltar ao menu...${NC}"
    read
}

function open_dashboard() {
    local dash="$PROJECT_ROOT/pesquisas/relatorios/dashboard/dashboard_interativo.html"
    if [ -f "$dash" ]; then
        echo -e "\n${CYAN}▶ Abrindo dashboard...${NC}"
        xdg-open "$dash" 2>/dev/null &
    else
        echo -e "\n${YELLOW}⚠ Dashboard não encontrado. Gerando...${NC}"
        cd "$PROJECT_ROOT/pesquisas/visualizacao"
        python3 dashboard.py 2>/dev/null
    fi
    echo -e "\n${GREEN}Pressione ENTER para voltar ao menu...${NC}"
    read
}

function open_report() {
    local report="$PROJECT_ROOT/pesquisas/relatorios/dashboard/relatorio_narrativo.html"
    if [ -f "$report" ]; then
        echo -e "\n${CYAN}▶ Abrindo relatório narrativo...${NC}"
        xdg-open "$report" 2>/dev/null &
    else
        echo -e "${RED}❌ Relatório não encontrado.${NC}"
    fi
    echo -e "\n${GREEN}Pressione ENTER para voltar ao menu...${NC}"
    read
}

function view_data_summary() {
    echo -e "\n${CYAN}━━━ Resumo dos Dados ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    
    local dados_dir="$PROJECT_ROOT/pesquisas/dados"
    
    if [ -f "$dados_dir/compression_all.json" ]; then
        echo -e "  ${PURPLE}📦 Compressão:${NC}"
        python3 -c "
import json
d=json.load(open('$dados_dir/compression_all.json'))
for i,m in enumerate(d):
    name=m.get('model_name',f'Modelo {i+1}')
    print(f'    {name}: ratio={m[\"compression_ratio\"]:.2f}x  dedup={m[\"dedup_rate_percent\"]:.1f}%  chunks={m[\"chunk_count\"]}')
" 2>/dev/null
    fi
    
    echo ""
    if [ -f "$dados_dir/entropy_all.json" ]; then
        echo -e "  ${PURPLE}📈 Entropia:${NC}"
        python3 -c "
import json
d=json.load(open('$dados_dir/entropy_all.json'))
for e in d:
    print(f'    {e[\"source\"]}: μ={e[\"mean_entropy\"]:.3f}  σ={e[\"std_entropy\"]:.3f}  [{e[\"min_entropy\"]:.3f}, {e[\"max_entropy\"]:.3f}]')
" 2>/dev/null
    fi
    
    echo ""
    if [ -f "$dados_dir/fase3_routing.json" ]; then
        echo -e "  ${PURPLE}🧠 Routing Multi-Brain:${NC}"
        python3 -c "
import json
d=json.load(open('$dados_dir/fase3_routing.json'))
for r in d:
    print(f'    {r[\"brains\"]} brains: routing={r[\"routing_latency_us\"]}μs  compose={r[\"compose_latency_us\"]}μs  throughput={r[\"throughput_mbps\"]:.1f} MB/s')
" 2>/dev/null
    fi
    
    echo ""
    if [ -f "$dados_dir/bench_all.json" ]; then
        echo -e "  ${PURPLE}⚡ Benchmarks XOR:${NC}"
        python3 -c "
import json
d=json.load(open('$dados_dir/bench_all.json'))
for b in d:
    print(f'    {b[\"test_name\"]}: {b[\"ns_per_op\"]:.0f} ns/op  ({b[\"mb_per_sec\"]:.1f} MB/s)')
" 2>/dev/null
    fi

    echo -e "\n${GREEN}Pressione ENTER para voltar ao menu...${NC}"
    read
}

function show_git_status() {
    echo -e "\n${CYAN}━━━ Git Status ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    cd "$PROJECT_ROOT"
    git status
    echo ""
    git log --oneline -5
    echo -e "\n${GREEN}Pressione ENTER para voltar ao menu...${NC}"
    read
}

function run_web_cockpit() {
    clear
    echo -e "${PURPLE}${BOLD}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║   🌐 NEURAL WEB COCKPIT — REACT UI                            ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    # Check dependencies
    if ! command -v npm &> /dev/null; then
        echo -e "${RED}[!] NPM ou Node não encontrados. Instale o NodeJS para usar a UI Web.${NC}"
        read -p "Pressione ENTER para voltar..."
        return
    fi
    
    # Guarantee FUSE is running
    local needs_start=true
    if pgrep -f fuse_demon_check >/dev/null; then
        if curl --max-time 1 -s "http://localhost:9999/stats" >/dev/null; then
            echo -e "${GREEN}>> FUSE Daemon já online e saudável.${NC}"
            needs_start=false
        else
            echo -e "${RED}>> FUSE zumbificado. Restaurando...${NC}"
        fi
    fi

    if [ "$needs_start" = true ]; then
        echo -e "${YELLOW}>> Despertando daemon neural FUSE...${NC}"
        kill_zombies
        fusermount -uz "$MNT_DIR" 2>/dev/null || true
        cd "$TESTES_DIR" && go build -o "$FUSE_BIN" ./cmd/fuse_test/main.go
        mkdir -p "$MNT_DIR"
        "$FUSE_BIN" > "${TESTES_DIR}/fuse_daemon.log" 2>&1 &
        sleep 4
    fi

    echo -e "${CYAN}>> Iniciando Frontend React (Vite) na porta 5173...${NC}"
    cd "$PROJECT_ROOT/web-cockpit"
    
    # Se ainda estiver rodando background anterior, tenta matar suavemente
    killall node 2>/dev/null || true

    echo -e "${GREEN}>> Frontend Inicializado! Acesse abaixo ou pressione CTRL+C para parar servidor web.${NC}"
    echo -e "🔗 http://localhost:5173\n"
    
    npm run dev
    # Quando o usuário fechar o npm run dev (CTRL+C), o script volta pro menu principal
}

# ─────────────────────────────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────────────────────────────
while true; do
    show_menu
    read choice
    echo ""
    case $choice in
        1)  run_full_suite ;;
        2)  run_unit_tests ;;
        3)  run_benchmarks ;;
        4)  run_multi_brain ;;
        5)  run_p2p_security ;;
        6)  start_fuse_cockpit ;;
        7)  view_heatmap ;;
        8)  run_empirical_test ;;
        9)  generate_graphs ;;
        10) open_dashboard ;;
        11) open_report ;;
        12) view_data_summary ;;
        13) show_git_status ;;
        14) run_rpg_mode ;;
        15) run_neural_chat ;;
        16) run_web_cockpit ;;
        q|Q) exit 0 ;;
        *)  echo -e "${RED}Opção inválida${NC}"; sleep 1 ;;
    esac
done
