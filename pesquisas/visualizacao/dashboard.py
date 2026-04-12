#!/usr/bin/env python3
"""
dashboard.py — Dashboard interativo com Plotly para análise dos resultados
Crompressor-Neurônio | Laboratório de Pesquisa

Uso:
    cd pesquisas/visualizacao
    pip install -r requirements.txt
    python dashboard.py

Abre um servidor HTTP local com dashboard interativo.
"""

import json
import sys
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    import numpy as np
except ImportError:
    print("❌ Dependências não instaladas!")
    print("   pip install -r requirements.txt")
    sys.exit(1)

# Diretórios
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DADOS_DIR = PROJECT_ROOT / "pesquisas" / "dados"
OUTPUT_DIR = PROJECT_ROOT / "pesquisas" / "relatorios" / "dashboard"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Paleta do Crompressor
CROM_COLORS = ['#9d4edd', '#00d2ff', '#00ff88', '#ff6b6b', '#ffd93d', '#6c5ce7']


def load_json(filename):
    """Carrega arquivo JSON."""
    filepath = DADOS_DIR / filename
    if not filepath.exists():
        return None
    with open(filepath) as f:
        return json.load(f)


def create_dashboard():
    """Cria dashboard completo interativo."""
    comp = load_json("compression_all.json")
    entropy = load_json("entropy_all.json")
    routing = load_json("routing_all.json")

    # Dashboard principal
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Compression Ratio por Modelo',
            'Entropia de Shannon (Distribuição)',
            'Taxa de Deduplicação',
            'Codebook Size vs Chunks',
            'Multi-Brain Routing Latency',
            'Memória por Neurônio',
        ],
        specs=[
            [{"type": "bar"}, {"type": "box"}],
            [{"type": "pie"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "bar"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # 1. Compression Ratio
    if comp:
        labels = [f"Modelo {i+1}" for i in range(len(comp))]
        ratios = [d['compression_ratio'] for d in comp]
        fig.add_trace(
            go.Bar(x=labels, y=ratios, marker_color=CROM_COLORS[:len(ratios)],
                   text=[f'{r:.2f}x' for r in ratios], textposition='outside',
                   name='Ratio'),
            row=1, col=1
        )
        fig.add_hline(y=3.0, line_dash="dash", line_color="#ff6b6b",
                      annotation_text="Threshold 3x", row=1, col=1)

    # 2. Entropy Distribution
    if entropy:
        for i, ent in enumerate(entropy):
            fig.add_trace(
                go.Box(y=ent['chunk_entropies'],
                       name=ent.get('source', f'modelo_{i}'),
                       marker_color=CROM_COLORS[i % len(CROM_COLORS)],
                       boxmean='sd'),
                row=1, col=2
            )

    # 3. Dedup Rate (Pie)
    if comp:
        avg_dedup = np.mean([d['dedup_rate_percent'] for d in comp])
        fig.add_trace(
            go.Pie(labels=['Deduplicado', 'Único'],
                   values=[avg_dedup, 100-avg_dedup],
                   marker_colors=['#9d4edd', '#333333'],
                   hole=0.4,
                   textinfo='percent+label'),
            row=2, col=1
        )

    # 4. Codebook vs Chunks
    if comp:
        fig.add_trace(
            go.Scatter(
                x=[d['chunk_count'] for d in comp],
                y=[d['codebook_size'] for d in comp],
                mode='markers+text',
                marker=dict(size=20, color=CROM_COLORS[:len(comp)], line=dict(width=2, color='white')),
                text=[f'M{i+1}' for i in range(len(comp))],
                textposition='top center',
                name='Codebook',
            ),
            row=2, col=2
        )

    # 5. Routing Latency
    if routing:
        fig.add_trace(
            go.Scatter(
                x=[d['num_brains'] for d in routing],
                y=[d['decision_time_ns'] for d in routing],
                mode='lines+markers',
                line=dict(color='#00d2ff', width=3),
                marker=dict(size=12, color='#00d2ff'),
                fill='tonexty',
                name='Routing Time',
            ),
            row=3, col=1
        )

    # 6. Memory per Brain
    if routing:
        fig.add_trace(
            go.Bar(
                x=[d['num_brains'] for d in routing],
                y=[d['memory_used_mb'] for d in routing],
                marker_color=CROM_COLORS[:len(routing)],
                text=[f'{d["memory_used_mb"]:.1f}MB' for d in routing],
                textposition='outside',
                name='Memory',
            ),
            row=3, col=2
        )

    # Layout
    fig.update_layout(
        title=dict(
            text='🧬 Crompressor-Neurônio — Dashboard de Pesquisa Interativo',
            font=dict(size=24, color='white'),
        ),
        template='plotly_dark',
        height=1200,
        showlegend=False,
        font=dict(family='Inter, sans-serif', color='white'),
        paper_bgcolor='#0d0d0d',
        plot_bgcolor='#1a1a2e',
    )

    # Salvar HTML
    outpath = OUTPUT_DIR / "dashboard_interativo.html"
    fig.write_html(str(outpath), include_plotlyjs='cdn')
    print(f"  ✓ Dashboard salvo: {outpath}")

    # Também mostrar no browser
    fig.show()

    return fig


def create_delta_deep_dive():
    """Dashboard detalhado dos tensores delta."""
    delta_files = sorted(DADOS_DIR.glob("deltas_*.json"))
    if not delta_files:
        print("  ⚠ Nenhum arquivo deltas_*.json encontrado")
        return

    all_deltas = []
    for f in delta_files:
        with open(f) as fh:
            deltas = json.load(fh)
            for d in deltas:
                d['model'] = f.stem.replace('deltas_', '')
                all_deltas.append(d)

    # Separar XOR e VQ
    xor_deltas = [d for d in all_deltas if d['delta_type'] == 'xor']

    if not xor_deltas:
        return

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['Delta Size (KB)', 'Sparsity vs Ratio', 'Latência (μs)']
    )

    # Delta Size
    fig.add_trace(
        go.Bar(
            x=list(range(len(xor_deltas))),
            y=[d['delta_size_bytes']/1024 for d in xor_deltas],
            marker_color='#9d4edd',
            name='Size',
        ),
        row=1, col=1
    )

    # Sparsity vs Ratio
    fig.add_trace(
        go.Scatter(
            x=[d['delta_brain_ratio_percent'] for d in xor_deltas],
            y=[d.get('sparsity_percent', 0) for d in xor_deltas],
            mode='markers',
            marker=dict(size=15, color='#00d2ff', line=dict(width=2, color='white')),
            name='XOR',
        ),
        row=1, col=2
    )

    # Latency
    fig.add_trace(
        go.Bar(
            x=list(range(len(xor_deltas))),
            y=[d['apply_latency_ns']/1000 for d in xor_deltas],
            marker_color='#00ff88',
            name='Latency',
        ),
        row=1, col=3
    )

    fig.update_layout(
        title='⚡ Tensor Delta — Análise Detalhada',
        template='plotly_dark',
        height=500,
        showlegend=False,
        paper_bgcolor='#0d0d0d',
        plot_bgcolor='#1a1a2e',
    )

    outpath = OUTPUT_DIR / "delta_deep_dive.html"
    fig.write_html(str(outpath), include_plotlyjs='cdn')
    print(f"  ✓ Delta Deep Dive: {outpath}")


def main():
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║   🎯 DASHBOARD INTERATIVO — Crompressor-Neurônio         ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()

    if not DADOS_DIR.exists() or not any(DADOS_DIR.glob("*.json")):
        print("❌ Nenhum dado encontrado em pesquisas/dados/")
        print("   Execute primeiro: pesquisas/scripts/run_all_tests.sh")
        sys.exit(1)

    create_dashboard()
    create_delta_deep_dive()

    print()
    print(f"✅ Dashboards salvos em: {OUTPUT_DIR}")
    print("   Abra os arquivos .html no navegador para interatividade")


if __name__ == "__main__":
    main()
