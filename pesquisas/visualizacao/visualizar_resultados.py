#!/usr/bin/env python3
"""
visualizar_resultados.py — Gera gráficos estáticos a partir dos dados JSON
Crompressor-Neurônio | Laboratório de Pesquisa

Uso:
    cd pesquisas/visualizacao
    pip install -r requirements.txt
    python visualizar_resultados.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo
import numpy as np

# Diretórios
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DADOS_DIR = PROJECT_ROOT / "pesquisas" / "dados"
OUTPUT_DIR = PROJECT_ROOT / "pesquisas" / "relatorios" / "graficos"

# Criar diretório de saída
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Estilo global
plt.style.use('dark_background')
COLORS = ['#9d4edd', '#00d2ff', '#00ff88', '#ff6b6b', '#ffd93d', '#6c5ce7']


def load_json(filename):
    """Carrega arquivo JSON do diretório de dados."""
    filepath = DADOS_DIR / filename
    if not filepath.exists():
        print(f"  ⚠ Arquivo não encontrado: {filepath}")
        return None
    with open(filepath) as f:
        return json.load(f)


def plot_compression_ratio():
    """Gráfico 1: Compression Ratio por modelo (bar chart)."""
    data = load_json("compression_all.json")
    if not data:
        return

    labels = [f"Modelo {i+1}\n({d['chunk_count']} chunks)" for i, d in enumerate(data)]
    ratios = [d['compression_ratio'] for d in data]
    dedup = [d['dedup_rate_percent'] for d in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Métricas de Compressão — Crompressor-Neurônio', fontsize=16, color='white')

    # Bar chart: ratio
    bars = ax1.bar(labels, ratios, color=COLORS[:len(ratios)], edgecolor='white', linewidth=0.5)
    ax1.set_ylabel('Compression Ratio (x)', fontsize=12)
    ax1.set_title('Ratio de Compressão por Modelo', fontsize=13)
    ax1.axhline(y=3.0, color='#ff6b6b', linestyle='--', alpha=0.7, label='Threshold (3x)')
    ax1.legend()
    for bar, ratio in zip(bars, ratios):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{ratio:.2f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Bar chart: dedup rate
    bars2 = ax2.bar(labels, dedup, color=COLORS[:len(dedup)], edgecolor='white', linewidth=0.5)
    ax2.set_ylabel('Deduplication Rate (%)', fontsize=12)
    ax2.set_title('Taxa de Deduplicação por Modelo', fontsize=13)
    for bar, rate in zip(bars2, dedup):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    outpath = OUTPUT_DIR / "01_compression_ratio.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {outpath}")


def plot_entropy_distribution():
    """Gráfico 2: Distribuição de entropia por modelo (box plot)."""
    data = load_json("entropy_all.json")
    if not data:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Distribuição de Entropia de Shannon', fontsize=16, color='white')

    labels = [d.get('source', f'modelo_{i}') for i, d in enumerate(data)]
    entropies = [d['chunk_entropies'] for d in data]

    bp = ax.boxplot(entropies, labels=labels, patch_artist=True, notch=True)
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Shannon Entropy (bits/byte)', fontsize=12)
    ax.set_title('Entropia por Chunk — Todas as Vertentes', fontsize=13)
    ax.axhline(y=8.0, color='#ff6b6b', linestyle='--', alpha=0.5, label='Máximo teórico (8 bits/byte)')
    ax.legend()

    # Anotar médias
    for i, ent in enumerate(entropies):
        mean_val = np.mean(ent)
        ax.annotate(f'μ={mean_val:.3f}', xy=(i+1, mean_val),
                    fontsize=9, ha='center', color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    plt.tight_layout()
    outpath = OUTPUT_DIR / "02_entropy_distribution.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {outpath}")


def plot_delta_analysis():
    """Gráfico 3: Análise de deltas — tamanho e esparsificação."""
    # Procurar todos os arquivos deltas_*.json
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

    if not all_deltas:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Análise de Tensor Delta — XOR vs VQ', fontsize=16, color='white')

    # Separar XOR e VQ
    xor_deltas = [d for d in all_deltas if d['delta_type'] == 'xor']
    vq_deltas = [d for d in all_deltas if d['delta_type'] == 'vq']

    # Plot 1: Delta Size (bar)
    ax = axes[0]
    if xor_deltas:
        sizes = [d['delta_size_bytes'] / 1024 for d in xor_deltas]  # KB
        sparsities = [d.get('sparsity_percent', 0) for d in xor_deltas]
        x = range(len(sizes))
        ax.bar(x, sizes, color='#9d4edd', alpha=0.8, label='XOR')
        ax.set_xlabel('Delta Index')
        ax.set_ylabel('Tamanho (KB)')
        ax.set_title('Tamanho dos Deltas XOR')
        ax.legend()

    # Plot 2: Sparsity (scatter)
    ax = axes[1]
    if xor_deltas:
        ratios = [d['delta_brain_ratio_percent'] for d in xor_deltas]
        sparsities = [d.get('sparsity_percent', 0) for d in xor_deltas]
        ax.scatter(ratios, sparsities, c='#00d2ff', s=100, alpha=0.8, edgecolors='white')
        ax.set_xlabel('Delta/Brain Ratio (%)')
        ax.set_ylabel('Sparsity (%)')
        ax.set_title('Ratio vs Esparsificação')
        ax.axhline(y=80, color='#00ff88', linestyle='--', alpha=0.5, label='Alvo: 80%')
        ax.axvline(x=5, color='#ff6b6b', linestyle='--', alpha=0.5, label='Alvo: <5%')
        ax.legend()

    # Plot 3: Latência (bar)
    ax = axes[2]
    if xor_deltas:
        latencies = [d['apply_latency_ns'] / 1000 for d in xor_deltas]  # μs
        x = range(len(latencies))
        ax.bar(x, latencies, color='#00ff88', alpha=0.8)
        ax.set_xlabel('Delta Index')
        ax.set_ylabel('Latência (μs)')
        ax.set_title('Latência de Aplicação XOR')
        ax.axhline(y=10, color='#ff6b6b', linestyle='--', alpha=0.5, label='Alvo: <10μs')
        ax.legend()

    plt.tight_layout()
    outpath = OUTPUT_DIR / "03_delta_analysis.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {outpath}")


def plot_routing():
    """Gráfico 4: Multi-Brain Routing — decisão e memória."""
    data = load_json("routing_all.json")
    if not data:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Multi-Brain Routing — Performance', fontsize=16, color='white')

    num_brains = [d['num_brains'] for d in data]
    decision_times = [d['decision_time_ns'] for d in data]
    memory = [d['memory_used_mb'] for d in data]

    # Decision time
    ax1.plot(num_brains, decision_times, 'o-', color='#9d4edd', linewidth=2, markersize=10)
    ax1.set_xlabel('Número de Neurônios', fontsize=12)
    ax1.set_ylabel('Tempo de Decisão (ns)', fontsize=12)
    ax1.set_title('Latência de Routing vs N Neurônios', fontsize=13)
    ax1.fill_between(num_brains, decision_times, alpha=0.2, color='#9d4edd')

    # Memory
    ax2.bar(num_brains, memory, color=COLORS[:len(memory)], edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('Número de Neurônios', fontsize=12)
    ax2.set_ylabel('Memória Utilizada (MB)', fontsize=12)
    ax2.set_title('RAM por Neurônio Adicional', fontsize=13)

    plt.tight_layout()
    outpath = OUTPUT_DIR / "04_routing_performance.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {outpath}")


def plot_summary_dashboard():
    """Gráfico 5: Dashboard resumo com todas as métricas."""
    comp = load_json("compression_all.json")
    entropy = load_json("entropy_all.json")
    routing = load_json("routing_all.json")

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('🧬 Crompressor-Neurônio — Dashboard de Pesquisa', fontsize=20, color='white', fontweight='bold')

    # Grid 2x3
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # 1. Compression Ratio
    if comp:
        ax = fig.add_subplot(gs[0, 0])
        ratios = [d['compression_ratio'] for d in comp]
        bars = ax.barh(range(len(ratios)), ratios, color=COLORS[:len(ratios)])
        ax.set_yticks(range(len(ratios)))
        ax.set_yticklabels([f'M{i+1}' for i in range(len(ratios))])
        ax.set_xlabel('Ratio (x)')
        ax.set_title('Compression Ratio', fontsize=12, fontweight='bold')
        ax.axvline(x=3.0, color='#ff6b6b', linestyle='--', alpha=0.7)

    # 2. Entropy Mean
    if entropy:
        ax = fig.add_subplot(gs[0, 1])
        means = [d['mean_entropy'] for d in entropy]
        stds = [d['std_entropy'] for d in entropy]
        labels = [d.get('source', f'm{i}') for i, d in enumerate(entropy)]
        ax.bar(labels, means, yerr=stds, color=COLORS[:len(means)], capsize=5, alpha=0.8)
        ax.set_ylabel('bits/byte')
        ax.set_title('Entropia Média (±σ)', fontsize=12, fontweight='bold')

    # 3. Routing Decision Time
    if routing:
        ax = fig.add_subplot(gs[0, 2])
        n = [d['num_brains'] for d in routing]
        t = [d['decision_time_ns'] for d in routing]
        ax.plot(n, t, 'o-', color='#00d2ff', linewidth=2, markersize=8)
        ax.fill_between(n, t, alpha=0.2, color='#00d2ff')
        ax.set_xlabel('Neurônios')
        ax.set_ylabel('ns')
        ax.set_title('Routing Latency', fontsize=12, fontweight='bold')

    # 4. Dedup Rate (pie chart)
    if comp:
        ax = fig.add_subplot(gs[1, 0])
        avg_dedup = np.mean([d['dedup_rate_percent'] for d in comp])
        ax.pie([avg_dedup, 100-avg_dedup],
               labels=['Dedup', 'Único'],
               colors=['#9d4edd', '#333333'],
               autopct='%1.1f%%',
               startangle=90,
               textprops={'color': 'white'})
        ax.set_title('Taxa de Deduplicação Média', fontsize=12, fontweight='bold')

    # 5. Codebook Size
    if comp:
        ax = fig.add_subplot(gs[1, 1])
        cb_sizes = [d['codebook_size'] for d in comp]
        chunk_counts = [d['chunk_count'] for d in comp]
        ax.scatter(chunk_counts, cb_sizes, c=COLORS[:len(cb_sizes)], s=200, edgecolors='white', linewidth=1.5)
        ax.set_xlabel('Total Chunks')
        ax.set_ylabel('Codebook Size')
        ax.set_title('Codebook vs Chunks', fontsize=12, fontweight='bold')

    # 6. Memory Usage
    if routing:
        ax = fig.add_subplot(gs[1, 2])
        n = [d['num_brains'] for d in routing]
        m = [d['memory_used_mb'] for d in routing]
        ax.bar(n, m, color='#00ff88', alpha=0.8, edgecolor='white')
        ax.set_xlabel('Neurônios')
        ax.set_ylabel('MB')
        ax.set_title('Memória por Brain', fontsize=12, fontweight='bold')

    outpath = OUTPUT_DIR / "05_dashboard_completo.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {outpath}")


def main():
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║   📊 VISUALIZADOR DE RESULTADOS — Crompressor-Neurônio   ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()

    if not DADOS_DIR.exists() or not any(DADOS_DIR.glob("*.json")):
        print("❌ Nenhum dado encontrado em pesquisas/dados/")
        print("   Execute primeiro: pesquisas/scripts/run_all_tests.sh")
        sys.exit(1)

    print(f"  Dados: {DADOS_DIR}")
    print(f"  Saída: {OUTPUT_DIR}")
    print()

    plot_compression_ratio()
    plot_entropy_distribution()
    plot_delta_analysis()
    plot_routing()
    plot_summary_dashboard()

    print()
    print(f"✅ Todos os gráficos salvos em: {OUTPUT_DIR}")
    print(f"   Total: {len(list(OUTPUT_DIR.glob('*.png')))} imagens")


if __name__ == "__main__":
    main()
