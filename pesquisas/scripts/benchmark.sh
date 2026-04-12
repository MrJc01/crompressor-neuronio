#!/bin/bash
# =============================================================================
# benchmark.sh — Benchmarks detalhados com go test -bench
# Crompressor-Neurônio | Laboratório de Pesquisa
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TESTES_DIR="$PROJECT_ROOT/pesquisas/testes"
DADOS_DIR="$PROJECT_ROOT/pesquisas/dados"

PURPLE='\033[0;35m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${PURPLE}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   ⚡ BENCHMARK DETALHADO — Crompressor-Neurônio           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

mkdir -p "$DADOS_DIR"
cd "$TESTES_DIR"

# =============================================================================
# Go Benchmarks nativos
# =============================================================================
echo -e "${BLUE}━━━ Go Benchmarks ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

BENCH_FILE="$DADOS_DIR/go_bench_$(date +%Y%m%d_%H%M%S).txt"

echo -e "  Executando go test -bench=. -benchmem ..."
go test -bench=. -benchmem -count=3 ./pkg/... 2>&1 | tee "$BENCH_FILE" || true

echo -e "\n  ${GREEN}✓${NC} Resultados salvos em: $BENCH_FILE"

# =============================================================================
# Benchmark de Memória
# =============================================================================
echo -e "\n${BLUE}━━━ Memory Profiling ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

MEMPROF="$DADOS_DIR/memprof_$(date +%Y%m%d_%H%M%S).out"

echo -e "  Executando com -memprofile ..."
go test -run=^$ -bench=. -memprofile="$MEMPROF" ./pkg/... 2>&1 || true

if [ -f "$MEMPROF" ]; then
    echo -e "  ${GREEN}✓${NC} Memory profile: $MEMPROF"
    echo -e "  Para analisar: go tool pprof $MEMPROF"
fi

# =============================================================================
# Benchmark de CPU
# =============================================================================
echo -e "\n${BLUE}━━━ CPU Profiling ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

CPUPROF="$DADOS_DIR/cpuprof_$(date +%Y%m%d_%H%M%S).out"

echo -e "  Executando com -cpuprofile ..."
go test -run=^$ -bench=. -cpuprofile="$CPUPROF" ./pkg/... 2>&1 || true

if [ -f "$CPUPROF" ]; then
    echo -e "  ${GREEN}✓${NC} CPU profile: $CPUPROF"
    echo -e "  Para analisar: go tool pprof $CPUPROF"
fi

# =============================================================================
# Sumário
# =============================================================================
echo -e "\n${PURPLE}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   ✅ BENCHMARKS COMPLETOS                                ║"
echo "║                                                          ║"
echo "║   Dados: pesquisas/dados/*bench* pesquisas/dados/*prof*  ║"
echo "║                                                          ║"
echo "║   Análise:                                               ║"
echo "║     go tool pprof pesquisas/dados/cpuprof_*.out          ║"
echo "║     go tool pprof pesquisas/dados/memprof_*.out          ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"
