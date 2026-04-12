#!/bin/bash
# =============================================================================
# run_all_tests.sh — Executa todos os testes Go e gera dados
# Crompressor-Neurônio | Laboratório de Pesquisa
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TESTES_DIR="$PROJECT_ROOT/pesquisas/testes"
DADOS_DIR="$PROJECT_ROOT/pesquisas/dados"
RELATORIOS_DIR="$PROJECT_ROOT/pesquisas/relatorios"

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   🧬 CROMPRESSOR-NEURÔNIO — Suite de Testes Completa     ║"
echo "║   Versão: V0.1 Exploratório                              ║"
echo "║   Data: $(date '+%Y-%m-%d %H:%M:%S')                     ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# =============================================================================
# 1. PREPARAÇÃO
# =============================================================================
echo -e "${BLUE}━━━ [1/5] Preparação ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

mkdir -p "$DADOS_DIR"
mkdir -p "$RELATORIOS_DIR"

# Verificar Go
if ! command -v go &> /dev/null; then
    echo -e "${RED}❌ Go não encontrado! Instale Go 1.22+ primeiro.${NC}"
    exit 1
fi

GO_VERSION=$(go version | awk '{print $3}')
echo -e "  ${GREEN}✓${NC} Go encontrado: $GO_VERSION"

# Verificar Python (para visualização posterior)
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 --version 2>&1)
    echo -e "  ${GREEN}✓${NC} Python encontrado: $PY_VERSION"
else
    echo -e "  ${YELLOW}⚠${NC} Python3 não encontrado. Visualização não será possível."
fi

echo -e "  ${GREEN}✓${NC} Diretório de dados: $DADOS_DIR"
echo -e "  ${GREEN}✓${NC} Diretório de relatórios: $RELATORIOS_DIR"

# =============================================================================
# 2. COMPILAÇÃO
# =============================================================================
echo -e "\n${BLUE}━━━ [2/5] Compilação ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

cd "$TESTES_DIR"

echo -e "  Baixando dependências..."
go mod tidy 2>/dev/null || true

echo -e "  Compilando run_all..."
go build -o "$TESTES_DIR/bin/run_all" ./cmd/run_all/
echo -e "  ${GREEN}✓${NC} Binário: $TESTES_DIR/bin/run_all"

# =============================================================================
# 3. EXECUÇÃO DOS TESTES
# =============================================================================
echo -e "\n${BLUE}━━━ [3/5] Execução dos Testes ━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

cd "$PROJECT_ROOT"
START_TIME=$(date +%s)

"$TESTES_DIR/bin/run_all"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo -e "\n  ${GREEN}✓${NC} Testes concluídos em ${DURATION}s"

# =============================================================================
# 4. VERIFICAÇÃO DOS DADOS
# =============================================================================
echo -e "\n${BLUE}━━━ [4/5] Verificação dos Dados Gerados ━━━━━━━━━━━━━━━━${NC}"

FILE_COUNT=$(find "$DADOS_DIR" -name "*.json" | wc -l)
TOTAL_SIZE=$(du -sh "$DADOS_DIR" 2>/dev/null | cut -f1)

echo -e "  📊 Arquivos JSON gerados: $FILE_COUNT"
echo -e "  💾 Tamanho total: $TOTAL_SIZE"

for f in "$DADOS_DIR"/*.json; do
    if [ -f "$f" ]; then
        BASENAME=$(basename "$f")
        SIZE=$(du -h "$f" | cut -f1)
        echo -e "     ${GREEN}✓${NC} $BASENAME ($SIZE)"
    fi
done

# =============================================================================
# 5. SUMÁRIO
# =============================================================================
echo -e "\n${PURPLE}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   ✅ SUITE DE TESTES COMPLETA                            ║"
echo "║                                                          ║"
echo "║   Dados:      pesquisas/dados/*.json                     ║"
echo "║   Relatórios: pesquisas/relatorios/                      ║"
echo "║   Duração:    ${DURATION}s                               ║"
echo "║                                                          ║"
echo "║   Próximos passos:                                       ║"
echo "║   1. ./scripts/generate_report.sh                        ║"
echo "║   2. cd pesquisas/visualizacao && python dashboard.py    ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"
