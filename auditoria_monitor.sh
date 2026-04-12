#!/bin/bash
# =============================================================================
# 🕵️ CROM-AUDIT: Teste automatizado do Neural Monitor
# Injeta inputs no monitor.sh para validar rotas, pausas e menus.
# =============================================================================

LOG="audit_log.txt"
> "$LOG"

echo "======================================================" | tee -a "$LOG"
echo " INICIANDO AUDITORIA DO MULTI-BRAIN MONITOR" | tee -a "$LOG"
echo "======================================================" | tee -a "$LOG"

# 1. Testando Dados JSON (Opção 12)
echo -e "\n[!] TESTE 1: Lendo Resumo JSON [Opção 12]" | tee -a "$LOG"
# Input: "12" -> "Enter" para fechar resumo -> "q" para sair
echo -e "12\n\nq\n" | ./monitor.sh | grep -E "Compressão:|Entropia:|Routing|Benchmarks XOR:" >> "$LOG" 2>&1

# 2. Testando Unit Tests (Opção 2)
echo -e "\n[!] TESTE 2: Rotina de Testes [Opção 2]" | tee -a "$LOG"
echo -e "2\n\nq\n" | ./monitor.sh | grep -E "PASS|FAIL" | tail -n 5 >> "$LOG" 2>&1

# 3. Testando Modo RPG (Opção 14)
echo -e "\n[!] TESTE 3: Jornada RPG (6 pausas) [Opção 14]" | tee -a "$LOG"
# Input: 14 -> 6 enters (um para cada etapa) -> 1 enter para voltar -> q para sair
echo -e "14\n\n\n\n\n\n\nq\n" | ./monitor.sh | grep -E "O GRANDE GARGALO|SEÇÃO 1|SEÇÃO 2|SEÇÃO 3|SEÇÃO 4|JORNADA CONCLUÍDA" >> "$LOG" 2>&1

# 4. Testando Chat Neural (Opção 15)
echo -e "\n[!] TESTE 4: Chat Neural [Opção 15]" | tee -a "$LOG"
# Input: 15 -> '1' (Escolher Base) -> 'Olá cérebro' -> 'sair' -> 'q' (Sair menu)
echo -e "15\n1\nOlá cérebro\nsair\nq\n" | ./monitor.sh | grep -E "CHAT NEURAL|Conectado ao modelo|Você: Olá|Cérebro:" >> "$LOG" 2>&1

# 5. Testando FUSE Cockpit (Opção 6)
echo -e "\n[!] TESTE 5: FUSE Cockpit Live Mode [Opção 6]" | tee -a "$LOG"
# Input: 6 -> 1(Persona) -> s(Prompt) -> q(Sair cockpit) -> \n(Voltar Menu) -> q(Sair app)
echo -e "6\n1\ns\nq\n\nq\n" | timeout 15 ./monitor.sh | grep -E "FUSE ONLINE|API REST|NEURAL COCKPIT" | head -n 5 >> "$LOG" 2>&1

echo -e "\n======================================================" | tee -a "$LOG"
echo " AUDITORIA CONCLUÍDA. VOU ANALISAR O ARQUIVO." | tee -a "$LOG"
echo "======================================================" | tee -a "$LOG"
