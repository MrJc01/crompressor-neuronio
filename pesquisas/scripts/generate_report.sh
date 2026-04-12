#!/bin/bash
# =============================================================================
# generate_report.sh — Gera relatórios agregados a partir dos dados JSON
# Crompressor-Neurônio | Laboratório de Pesquisa
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DADOS_DIR="$PROJECT_ROOT/pesquisas/dados"
RELATORIOS_DIR="$PROJECT_ROOT/pesquisas/relatorios"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${PURPLE}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   📋 GERADOR DE RELATÓRIOS — Crompressor-Neurônio        ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

mkdir -p "$RELATORIOS_DIR"

REPORT_FILE="$RELATORIOS_DIR/relatorio_$(date +%Y%m%d_%H%M%S).md"

# =============================================================================
# Header do Relatório
# =============================================================================
cat > "$REPORT_FILE" << 'HEADER'
# 📊 Relatório de Pesquisa — Crompressor-Neurônio

> Gerado automaticamente por `generate_report.sh`

---

HEADER

echo "Data: $(date '+%Y-%m-%d %H:%M:%S')" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# =============================================================================
# Seção 1: Compressão
# =============================================================================
echo "## 1. Métricas de Compressão" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

if [ -f "$DADOS_DIR/compression_all.json" ]; then
    echo '```json' >> "$REPORT_FILE"
    cat "$DADOS_DIR/compression_all.json" >> "$REPORT_FILE"
    echo '```' >> "$REPORT_FILE"
    echo -e "  ${GREEN}✓${NC} Compressão adicionada ao relatório"
else
    echo "_Dados de compressão não encontrados. Execute run_all_tests.sh primeiro._" >> "$REPORT_FILE"
    echo -e "  ${RED}✗${NC} compression_all.json não encontrado"
fi
echo "" >> "$REPORT_FILE"

# =============================================================================
# Seção 2: Entropia
# =============================================================================
echo "## 2. Métricas de Entropia de Shannon" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

if [ -f "$DADOS_DIR/entropy_all.json" ]; then
    # Extrair apenas resumos (sem os arrays de chunk entropies)
    echo '```json' >> "$REPORT_FILE"
    python3 -c "
import json, sys
with open('$DADOS_DIR/entropy_all.json') as f:
    data = json.load(f)
for item in data:
    item.pop('chunk_entropies', None)
json.dump(data, sys.stdout, indent=2)
" >> "$REPORT_FILE" 2>/dev/null || cat "$DADOS_DIR/entropy_all.json" >> "$REPORT_FILE"
    echo '```' >> "$REPORT_FILE"
    echo -e "  ${GREEN}✓${NC} Entropia adicionada ao relatório"
else
    echo "_Dados de entropia não encontrados._" >> "$REPORT_FILE"
fi
echo "" >> "$REPORT_FILE"

# =============================================================================
# Seção 3: Deltas
# =============================================================================
echo "## 3. Métricas de Tensor Delta" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

for f in "$DADOS_DIR"/deltas_*.json; do
    if [ -f "$f" ]; then
        BASENAME=$(basename "$f" .json)
        echo "### $BASENAME" >> "$REPORT_FILE"
        echo '```json' >> "$REPORT_FILE"
        cat "$f" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    fi
done
echo -e "  ${GREEN}✓${NC} Deltas adicionados ao relatório"

# =============================================================================
# Seção 4: Benchmarks
# =============================================================================
echo "## 4. Benchmarks (ns/op)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

for f in "$DADOS_DIR"/bench_*.json; do
    if [ -f "$f" ]; then
        BASENAME=$(basename "$f" .json)
        echo "### $BASENAME" >> "$REPORT_FILE"
        echo '```json' >> "$REPORT_FILE"
        cat "$f" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    fi
done
echo -e "  ${GREEN}✓${NC} Benchmarks adicionados ao relatório"

# =============================================================================
# Seção 5: Routing
# =============================================================================
echo "## 5. Multi-Brain Routing" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

if [ -f "$DADOS_DIR/routing_all.json" ]; then
    echo '```json' >> "$REPORT_FILE"
    cat "$DADOS_DIR/routing_all.json" >> "$REPORT_FILE"
    echo '```' >> "$REPORT_FILE"
    echo -e "  ${GREEN}✓${NC} Routing adicionado ao relatório"
else
    echo "_Dados de routing não encontrados._" >> "$REPORT_FILE"
fi
echo "" >> "$REPORT_FILE"

# =============================================================================
# Footer
# =============================================================================
cat >> "$REPORT_FILE" << 'FOOTER'

---

## Análise & Próximos Passos

> Para visualização interativa:
> ```bash
> cd pesquisas/visualizacao
> pip install -r requirements.txt
> python dashboard.py
> ```

> Para gráficos estáticos:
> ```bash
> python pesquisas/visualizacao/visualizar_resultados.py
> ```

---

*Relatório gerado automaticamente pelo Crompressor-Neurônio Lab*
FOOTER

echo -e "\n${GREEN}✅ Relatório gerado: $REPORT_FILE${NC}"
echo -e "   Tamanho: $(du -h "$REPORT_FILE" | cut -f1)"
