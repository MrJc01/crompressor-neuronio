#!/usr/bin/env python3
"""
LAB07 — Delta Branch Store
===========================
Benchmark puro de memória: Delta vs Cópia Completa para arrays grandes.
- Base state: array numpy de 1MB
- Branches: apenas índices e valores que diferem
- Operações: criar, ler, merge, delete, colapso

Saída: JSON em pesquisa0/resultados/lab07_results.json
"""

import json
import os
import sys
import time
import random
from datetime import datetime
from typing import Dict, List, Optional, Any

SEED = 42
random.seed(SEED)

RESULTADOS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'resultados')

# Tentar usar numpy, fallback para lista pura
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("  ⚠ numpy não disponível, usando listas Python puras")


# ─── Delta Branch Store ──────────────────────────────────────
class DeltaBranchStore:
    """Armazena branches como Deltas sobre um estado base."""

    def __init__(self, base_state):
        if HAS_NUMPY:
            self.base = np.array(base_state, dtype=np.float32)
        else:
            self.base = list(base_state)
        self.branches: Dict[str, Dict[int, float]] = {}  # branch_id → {index: value}
        self.base_size_bytes = len(self.base) * 4  # float32

    def criar_branch(self, branch_id: str, divergencia_pct: float = 0.01):
        """Cria branch modificando divergencia_pct dos valores."""
        n_mudancas = max(1, int(len(self.base) * divergencia_pct))
        indices = random.sample(range(len(self.base)), n_mudancas)
        delta = {}
        for idx in indices:
            if HAS_NUMPY:
                delta[idx] = float(self.base[idx] + random.gauss(0, 0.1))
            else:
                delta[idx] = self.base[idx] + random.gauss(0, 0.1)
        self.branches[branch_id] = delta

    def ler(self, branch_id: str, index: int) -> float:
        """Lê valor da branch (fallback para base se não modificado)."""
        if branch_id in self.branches and index in self.branches[branch_id]:
            return self.branches[branch_id][index]
        if HAS_NUMPY:
            return float(self.base[index])
        return self.base[index]

    def ler_completo(self, branch_id: str):
        """Reconstrói estado completo da branch."""
        if HAS_NUMPY:
            estado = self.base.copy()
        else:
            estado = list(self.base)
        if branch_id in self.branches:
            for idx, val in self.branches[branch_id].items():
                estado[idx] = val
        return estado

    def deletar(self, branch_id: str):
        """Remove branch."""
        if branch_id in self.branches:
            del self.branches[branch_id]

    def colapsar(self, branch_vencedora: str):
        """Colapsa: promove branch vencedora a base, descarta todas as outras."""
        if branch_vencedora in self.branches:
            for idx, val in self.branches[branch_vencedora].items():
                self.base[idx] = val
        self.branches.clear()

    def memoria_delta_bytes(self, branch_id: str) -> int:
        """Calcula memória usada pela branch (apenas delta)."""
        if branch_id not in self.branches:
            return 0
        # Cada entrada: int (index, 28 bytes) + float (8 bytes) + overhead dict
        return len(self.branches[branch_id]) * 36 + 64  # overhead do dict

    def memoria_total_deltas(self) -> int:
        """Memória total de todas as branches."""
        return sum(self.memoria_delta_bytes(bid) for bid in self.branches)

    def memoria_copia_completa(self) -> int:
        """Memória se cada branch fosse cópia completa."""
        return self.base_size_bytes * len(self.branches)


class CopiaBranchStore:
    """Para comparação: armazena branches como cópia completa."""

    def __init__(self, base_state):
        if HAS_NUMPY:
            self.base = np.array(base_state, dtype=np.float32)
        else:
            self.base = list(base_state)
        self.branches = {}

    def criar_branch(self, branch_id: str, divergencia_pct: float = 0.01):
        if HAS_NUMPY:
            copia = self.base.copy()
        else:
            copia = list(self.base)
        n_mudancas = max(1, int(len(self.base) * divergencia_pct))
        indices = random.sample(range(len(self.base)), n_mudancas)
        for idx in indices:
            copia[idx] += random.gauss(0, 0.1)
        self.branches[branch_id] = copia

    def memoria_total(self) -> int:
        if HAS_NUMPY:
            return sum(b.nbytes for b in self.branches.values())
        return sum(sys.getsizeof(b) for b in self.branches.values())


# ─── Benchmarks ───────────────────────────────────────────────
def benchmark_criar_branches(tamanho_base: int, divergencias: List[float], n_branches: int = 100):
    """Benchmark: criar N branches com diferentes percentuais de divergência."""
    resultados = {}

    for div in divergencias:
        random.seed(SEED)
        if HAS_NUMPY:
            base = np.random.randn(tamanho_base).astype(np.float32)
        else:
            base = [random.gauss(0, 1) for _ in range(tamanho_base)]

        store_delta = DeltaBranchStore(base)
        store_copia = CopiaBranchStore(base)

        # Criar branches - Delta
        t0 = time.perf_counter()
        for i in range(n_branches):
            store_delta.criar_branch(f"d_{i}", div)
        t_delta = time.perf_counter() - t0

        # Criar branches - Cópia
        random.seed(SEED)
        t0 = time.perf_counter()
        for i in range(n_branches):
            store_copia.criar_branch(f"c_{i}", div)
        t_copia = time.perf_counter() - t0

        mem_delta = store_delta.memoria_total_deltas()
        mem_copia = store_copia.memoria_total()
        economia = (1 - mem_delta / max(1, mem_copia)) * 100

        resultados[f"div_{div}"] = {
            "divergencia_pct": div,
            "n_branches": n_branches,
            "tamanho_base_bytes": tamanho_base * 4,
            "memoria_delta_bytes": mem_delta,
            "memoria_copia_bytes": mem_copia,
            "economia_pct": round(economia, 2),
            "ratio": round(mem_copia / max(1, mem_delta), 2),
            "tempo_criar_delta_ms": round(t_delta * 1000, 2),
            "tempo_criar_copia_ms": round(t_copia * 1000, 2),
        }

    return resultados


def benchmark_colapso(tamanho_base: int, n_branches_list: List[int]):
    """Benchmark: tempo de colapso para diferentes números de branches."""
    resultados = {}

    for n in n_branches_list:
        if HAS_NUMPY:
            base = np.random.randn(tamanho_base).astype(np.float32)
        else:
            base = [random.gauss(0, 1) for _ in range(tamanho_base)]

        store = DeltaBranchStore(base)
        for i in range(n):
            store.criar_branch(f"b_{i}", 0.01)

        t0 = time.perf_counter()
        store.colapsar("b_0")
        t_colapso = time.perf_counter() - t0

        resultados[str(n)] = {
            "n_branches": n,
            "tempo_colapso_ms": round(t_colapso * 1000, 4),
            "tempo_colapso_us": round(t_colapso * 1_000_000, 2),
            "criterio_1ms": t_colapso * 1000 < 1.0
        }

    return resultados


def benchmark_leitura(tamanho_base: int, n_leituras: int = 10000):
    """Benchmark: velocidade de leitura Delta vs Cópia."""
    if HAS_NUMPY:
        base = np.random.randn(tamanho_base).astype(np.float32)
    else:
        base = [random.gauss(0, 1) for _ in range(tamanho_base)]

    store = DeltaBranchStore(base)
    store.criar_branch("test", 0.01)

    indices = [random.randint(0, tamanho_base - 1) for _ in range(n_leituras)]

    # Leitura Delta
    t0 = time.perf_counter()
    for idx in indices:
        store.ler("test", idx)
    t_delta = time.perf_counter() - t0

    # Leitura direta (baseline)
    estado = store.ler_completo("test")
    t0 = time.perf_counter()
    for idx in indices:
        _ = estado[idx]
    t_direto = time.perf_counter() - t0

    return {
        "n_leituras": n_leituras,
        "tempo_delta_ms": round(t_delta * 1000, 2),
        "tempo_direto_ms": round(t_direto * 1000, 2),
        "overhead_leitura_x": round(t_delta / max(0.0001, t_direto), 2),
        "latencia_delta_us": round(t_delta / n_leituras * 1_000_000, 4),
        "latencia_direto_us": round(t_direto / n_leituras * 1_000_000, 4),
    }


# ─── Main ─────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  LAB07 — DELTA BRANCH STORE")
    print(f"  Numpy disponível: {HAS_NUMPY}")
    print("=" * 70)

    tamanho_base = 256 * 1024  # 256K floats = ~1MB
    resultados = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "seed": SEED,
            "numpy": HAS_NUMPY,
            "tamanho_base_floats": tamanho_base,
            "tamanho_base_bytes": tamanho_base * 4
        },
        "experimentos": {}
    }

    # ── Benchmark 1: Memória vs Divergência ───────────────
    print("\n  ▶ Benchmark de memória: Delta vs Cópia (100 branches)...")
    divergencias = [0.0001, 0.001, 0.01, 0.1]
    mem = benchmark_criar_branches(tamanho_base, divergencias, n_branches=100)
    for key, dados in mem.items():
        print(f"    Div={dados['divergencia_pct']*100:>6.2f}%: "
              f"Delta={dados['memoria_delta_bytes']:>10,}B  "
              f"Cópia={dados['memoria_copia_bytes']:>12,}B  "
              f"Economia={dados['economia_pct']:>6.1f}%  "
              f"Ratio={dados['ratio']}x")
    resultados["experimentos"]["memoria_vs_divergencia"] = mem

    # ── Benchmark 2: Tempo de Colapso ─────────────────────
    print("\n  ▶ Benchmark de tempo de colapso...")
    colapso = benchmark_colapso(tamanho_base, [10, 50, 100, 500, 1000])
    for key, dados in colapso.items():
        status = "✅" if dados["criterio_1ms"] else "❌"
        print(f"    {dados['n_branches']:>5d} branches: "
              f"{dados['tempo_colapso_us']:>10.2f} μs  "
              f"({dados['tempo_colapso_ms']:.4f} ms)  {status} <1ms")
    resultados["experimentos"]["colapso"] = colapso

    # ── Benchmark 3: Leitura ──────────────────────────────
    print("\n  ▶ Benchmark de velocidade de leitura...")
    leitura = benchmark_leitura(tamanho_base, n_leituras=10000)
    print(f"    Delta:  {leitura['tempo_delta_ms']:.2f}ms  ({leitura['latencia_delta_us']:.4f} μs/read)")
    print(f"    Direto: {leitura['tempo_direto_ms']:.2f}ms  ({leitura['latencia_direto_us']:.4f} μs/read)")
    print(f"    Overhead: {leitura['overhead_leitura_x']}x")
    resultados["experimentos"]["leitura"] = leitura

    # ── Salvar ────────────────────────────────────────────
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTADOS_DIR, 'lab07_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Resultados salvos em: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
