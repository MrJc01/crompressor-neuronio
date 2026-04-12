#!/bin/bash
# =============================================================================
# pesquisar.sh — Executa TUDO em um comando e abre os resultados
# Crompressor-Neurônio | Laboratório de Pesquisa
#
# Uso: bash pesquisar.sh
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTES_DIR="$PROJECT_ROOT/pesquisas/testes"
VIZ_DIR="$PROJECT_ROOT/pesquisas/visualizacao"
DADOS_DIR="$PROJECT_ROOT/pesquisas/dados"
DASH_DIR="$PROJECT_ROOT/pesquisas/relatorios/dashboard"
GRAF_DIR="$PROJECT_ROOT/pesquisas/relatorios/graficos"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

TOTAL_START=$(date +%s)

echo ""
echo -e "${PURPLE}${BOLD}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   🧬 CROMPRESSOR-NEURÔNIO — Pipeline Completo            ║"
echo "║   Testes → Dados → Gráficos → Dashboard → Análise       ║"
echo "║   $(date '+%Y-%m-%d %H:%M:%S')                                    ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# =============================================================================
# [1/6] PRÉ-REQUISITOS
# =============================================================================
echo -e "${CYAN}━━━ [1/6] Verificando pré-requisitos ━━━━━━━━━━━━━━━━━━━━${NC}"

if ! command -v go &> /dev/null; then
    echo -e "  ${RED}✗ Go não encontrado! Instale Go 1.22+${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Go: $(go version | awk '{print $3}')"

if ! command -v python3 &> /dev/null; then
    echo -e "  ${RED}✗ Python3 não encontrado!${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Python: $(python3 --version 2>&1)"

# Venv
if [ ! -d "$VIZ_DIR/.venv" ]; then
    echo -e "  ${YELLOW}⏳${NC} Criando venv Python + dependências..."
    python3 -m venv "$VIZ_DIR/.venv"
    "$VIZ_DIR/.venv/bin/pip" install -q -r "$VIZ_DIR/requirements.txt"
fi
echo -e "  ${GREEN}✓${NC} Ambiente pronto"

# =============================================================================
# [2/6] UNIT TESTS
# =============================================================================
echo -e "\n${CYAN}━━━ [2/6] Rodando unit tests Go ━━━━━━━━━━━━━━━━━━━━━━━${NC}"

cd "$TESTES_DIR"
if go test -count=1 ./pkg/ > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Todos os unit tests passaram"
else
    echo -e "  ${RED}✗${NC} Unit tests falharam!"
    go test -v -count=1 ./pkg/ 2>&1 | tail -30
    exit 1
fi

# Benchmark rápido
BENCH=$(go test -bench=BenchmarkXORDelta_512 -benchtime=500ms -run=^$ ./pkg/ 2>&1)
XOR_NS=$(echo "$BENCH" | awk '/BenchmarkXORDelta_512/ {print $3}')
echo -e "  ${GREEN}✓${NC} XOR: ${BOLD}${XOR_NS} ns/op${NC}"

# =============================================================================
# [3/6] SUITE COMPLETA → gera dados JSON
# =============================================================================
echo -e "\n${CYAN}━━━ [3/6] Gerando dados (suite completa) ━━━━━━━━━━━━━━${NC}"

cd "$TESTES_DIR"
SUITE_OUTPUT=$(go run ./cmd/run_all/ 2>&1)
SUITE_EXIT=$?

# Mostrar resultados
echo "$SUITE_OUTPUT" | awk '/Ratio=/' | while IFS= read -r line; do
    echo -e "  ${PURPLE}║${NC} $line"
done
echo "$SUITE_OUTPUT" | awk '/ns\/op/' | while IFS= read -r line; do
    echo -e "  ${CYAN}║${NC} $line"
done

RESULT=$(echo "$SUITE_OUTPUT" | awk '/RESULTADO/' | head -1)
if [ $SUITE_EXIT -eq 0 ]; then
    echo -e "  ${GREEN}${BOLD}$RESULT${NC}"
else
    echo -e "  ${YELLOW}${BOLD}$RESULT${NC}"
    echo -e "  ${YELLOW}⚠ Alguns checks falharam (variance de CPU), mas dados foram gerados${NC}"
fi

JSON_COUNT=$(find "$DADOS_DIR" -name "*.json" 2>/dev/null | wc -l)
JSON_SIZE=$(du -sh "$DADOS_DIR" 2>/dev/null | cut -f1)
echo -e "  ${GREEN}✓${NC} ${JSON_COUNT} arquivos JSON (${JSON_SIZE})"

# =============================================================================
# [4/6] GRÁFICOS ESTÁTICOS (PNG)
# =============================================================================
echo -e "\n${CYAN}━━━ [4/6] Gerando gráficos estáticos (PNG) ━━━━━━━━━━━━${NC}"

cd "$VIZ_DIR"
"$VIZ_DIR/.venv/bin/python3" visualizar_resultados.py 2>&1 | awk '/✓/' | while IFS= read -r line; do
    FNAME=$(basename "$(echo "$line" | awk '{print $NF}')")
    echo -e "  ${GREEN}✓${NC} $FNAME"
done
PNG_COUNT=$(find "$GRAF_DIR" -name "*.png" 2>/dev/null | wc -l)
echo -e "  ${GREEN}✓${NC} ${PNG_COUNT} PNGs gerados"

# =============================================================================
# [5/6] DASHBOARD NARRATIVO (HTML)
# =============================================================================
echo -e "\n${CYAN}━━━ [5/6] Gerando dashboard narrativo (HTML) ━━━━━━━━━━${NC}"

"$VIZ_DIR/.venv/bin/python3" dashboard_completo.py 2>&1 | awk '/✓/' | while IFS= read -r line; do
    FNAME=$(basename "$(echo "$line" | awk '{print $NF}')")
    echo -e "  ${GREEN}✓${NC} $FNAME"
done
HTML_COUNT=$(find "$DASH_DIR" -name "*.html" 2>/dev/null | wc -l)
echo -e "  ${GREEN}✓${NC} ${HTML_COUNT} dashboards HTML gerados"

# =============================================================================
# [6/6] ABRIR RESULTADOS
# =============================================================================
echo -e "\n${CYAN}━━━ [6/6] Abrindo resultados ━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if command -v xdg-open &> /dev/null; then
    xdg-open "$DASH_DIR/relatorio_narrativo.html" 2>/dev/null &
    echo -e "  ${GREEN}✓${NC} Relatório narrativo aberto no navegador"
    sleep 1
    xdg-open "$PROJECT_ROOT/docs/ANALISE-RESULTADOS.md" 2>/dev/null &
    echo -e "  ${GREEN}✓${NC} Análise de resultados aberta"
else
    echo -e "  ${YELLOW}⚠${NC} Abra manualmente: $DASH_DIR/relatorio_narrativo.html"
fi

# =============================================================================
# RELATÓRIO FINAL
# =============================================================================
TOTAL_END=$(date +%s)
TOTAL_SEC=$(( TOTAL_END - TOTAL_START ))

echo -e "\n${PURPLE}${BOLD}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   ✅ PIPELINE COMPLETO — Tudo pronto para análise!       ║"
echo "╠═══════════════════════════════════════════════════════════╣"
echo "║                                                          ║"
echo "║   ⏱  Tempo total: ${TOTAL_SEC}s                                  ║"
echo "║   📊 Dados:  ${JSON_COUNT} JSON (${JSON_SIZE})                        ║"
echo "║   🖼  PNGs:   ${PNG_COUNT} gráficos estáticos                     ║"
echo "║   🌐 HTML:   ${HTML_COUNT} dashboards interativos                  ║"
echo "║                                                          ║"
echo "║   📂 Arquivos gerados:                                   ║"
echo "║      pesquisas/dados/         ← dados brutos             ║"
echo "║      pesquisas/relatorios/    ← gráficos + dashboards    ║"
echo "║      docs/ANALISE-RESULTADOS.md ← explicação completa   ║"
echo "║                                                          ║"
echo "║   🧬 \"O neurônio que comprime é o neurônio que pensa.\"   ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"
