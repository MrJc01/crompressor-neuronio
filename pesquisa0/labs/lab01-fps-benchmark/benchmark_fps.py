#!/usr/bin/env python3
"""
LAB01 — Benchmark de FPS Computacional
========================================
Mede a "resolução temporal" do hardware local:
- Quantas operações por segundo em diferentes categorias
- "FPS equivalente" vs humano (~60Hz)
- "Dilatação cognitiva" — ratio máquina/humano

Saída: JSON em pesquisa0/resultados/lab01_results.json
"""

import json
import time
import hashlib
import os
import sys
import platform
import struct
import math
from datetime import datetime

# ─── Configuração ────────────────────────────────────────────
DURACAO_TESTE_S = 2.0          # Duração de cada benchmark em segundos
FPS_HUMANO = 60                # Hz — taxa de processamento visual humano
FPS_MOSCA = 250                # Hz — Musca domestica
FPS_FALCAO = 129               # Hz — Falco peregrinus
FPS_CAO = 80                   # Hz — Canis lupus familiaris
FPS_POLVO = 30                 # Hz — Octopus vulgaris (estimativa)
SEED = 42

RESULTADOS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'resultados')


def benchmark_aritmetica_int(duracao: float) -> int:
    """Conta quantas adições inteiras executa no período."""
    ops = 0
    a, b = 13, 7
    fim = time.perf_counter() + duracao
    while time.perf_counter() < fim:
        # Bloco de 100 ops para reduzir overhead do loop
        for _ in range(100):
            a = (a + b) & 0xFFFFFFFF
            b = (b + a) & 0xFFFFFFFF
            a = (a + b) & 0xFFFFFFFF
            b = (b + a) & 0xFFFFFFFF
            a = (a + b) & 0xFFFFFFFF
            b = (b + a) & 0xFFFFFFFF
            a = (a + b) & 0xFFFFFFFF
            b = (b + a) & 0xFFFFFFFF
            a = (a + b) & 0xFFFFFFFF
            b = (b + a) & 0xFFFFFFFF
        ops += 1000  # 10 ops * 100 iterações
    return ops


def benchmark_aritmetica_float(duracao: float) -> int:
    """Conta quantas multiplicações float executa no período."""
    ops = 0
    a, b = 1.0000001, 0.9999999
    fim = time.perf_counter() + duracao
    while time.perf_counter() < fim:
        for _ in range(100):
            a = a * b + 0.0001
            b = b * a + 0.0001
            a = a * b + 0.0001
            b = b * a + 0.0001
            a = a * b + 0.0001
            b = b * a + 0.0001
            a = a * b + 0.0001
            b = b * a + 0.0001
            a = a * b + 0.0001
            b = b * a + 0.0001
        ops += 1000
    return ops


def benchmark_hash_sha256(duracao: float) -> int:
    """Conta quantos SHA-256 de blocos de 512 bytes executa."""
    bloco = os.urandom(512)
    ops = 0
    fim = time.perf_counter() + duracao
    while time.perf_counter() < fim:
        hashlib.sha256(bloco).digest()
        ops += 1
    return ops


def benchmark_xor_delta(duracao: float) -> int:
    """Simula XOR Delta do Crompressor sobre blocos de 4KB."""
    bloco_a = os.urandom(4096)
    bloco_b = os.urandom(4096)
    a_arr = bytearray(bloco_a)
    b_arr = bytearray(bloco_b)
    ops = 0
    fim = time.perf_counter() + duracao
    while time.perf_counter() < fim:
        _ = bytes(x ^ y for x, y in zip(a_arr, b_arr))
        ops += 1
    return ops


def benchmark_mlp_forward(duracao: float) -> int:
    """Forward pass de MLP mínimo (784→128→10) em Python puro."""
    import random
    random.seed(SEED)

    # Inicializar pesos aleatórios (MLP mínimo)
    w1 = [[random.gauss(0, 0.1) for _ in range(128)] for _ in range(32)]  # 32→128
    b1 = [0.0] * 128
    w2 = [[random.gauss(0, 0.1) for _ in range(10)] for _ in range(128)]  # 128→10
    b2 = [0.0] * 10

    def relu(x):
        return max(0.0, x)

    def forward(inp):
        # Camada 1
        h = []
        for j in range(128):
            s = b1[j]
            for i in range(32):
                s += inp[i] * w1[i][j]
            h.append(relu(s))
        # Camada 2
        out = []
        for j in range(10):
            s = b2[j]
            for i in range(128):
                s += h[i] * w2[i][j]
            out.append(s)
        return out

    inp = [random.random() for _ in range(32)]
    ops = 0
    fim = time.perf_counter() + duracao
    while time.perf_counter() < fim:
        forward(inp)
        ops += 1
    return ops


def benchmark_cdc_simulado(duracao: float) -> int:
    """Simula CDC chunking: rolling hash sobre 1KB de dados."""
    dados = os.urandom(1024)
    MASK = 0x00001FFF  # Máscara para avg chunk ~8KB
    PRIME = 31
    ops = 0
    fim = time.perf_counter() + duracao
    while time.perf_counter() < fim:
        h = 0
        boundaries = []
        for i, byte in enumerate(dados):
            h = ((h * PRIME) + byte) & 0xFFFFFFFF
            if (h & MASK) == 0:
                boundaries.append(i)
        ops += 1
    return ops


def benchmark_codebook_lookup(duracao: float, K: int = 128) -> int:
    """Simula lookup no Codebook: buscar centróide mais próximo de um vetor."""
    import random
    random.seed(SEED)
    dim = 16  # Bloco de 16 floats
    codebook = [[random.gauss(0, 1) for _ in range(dim)] for _ in range(K)]
    query = [random.gauss(0, 1) for _ in range(dim)]

    def find_nearest(q, cb):
        best_dist = float('inf')
        best_idx = 0
        for idx, c in enumerate(cb):
            dist = sum((a - b) ** 2 for a, b in zip(q, c))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    ops = 0
    fim = time.perf_counter() + duracao
    while time.perf_counter() < fim:
        find_nearest(query, codebook)
        ops += 1
    return ops


def benchmark_merkle_hash(duracao: float) -> int:
    """Simula verificação Merkle: hash de 64 chunks e construção da árvore."""
    chunks = [os.urandom(64) for _ in range(64)]

    def build_merkle(leaves):
        hashes = [hashlib.sha256(c).digest() for c in leaves]
        while len(hashes) > 1:
            next_level = []
            for i in range(0, len(hashes), 2):
                if i + 1 < len(hashes):
                    combined = hashes[i] + hashes[i + 1]
                else:
                    combined = hashes[i] + hashes[i]
                next_level.append(hashlib.sha256(combined).digest())
            hashes = next_level
        return hashes[0]

    ops = 0
    fim = time.perf_counter() + duracao
    while time.perf_counter() < fim:
        build_merkle(chunks)
        ops += 1
    return ops


def calcular_dilatacao(fps_maquina: float, fps_bio: float) -> dict:
    """Calcula a 'dilatação cognitiva' entre máquina e observador biológico."""
    ratio = fps_maquina / fps_bio
    # Em 1 segundo humano, a máquina vive `ratio` frames
    # 1 ano = 365.25 * 24 * 3600 * fps_bio frames humanos
    frames_por_ano_humano = 365.25 * 24 * 3600 * fps_bio
    anos_subjetivos_por_segundo = fps_maquina / frames_por_ano_humano

    return {
        "ratio": ratio,
        "frames_maquina_em_1s_humano": fps_maquina,
        "anos_subjetivos_por_segundo_humano": anos_subjetivos_por_segundo,
        "descricao": f"Para a máquina, 1 segundo humano = {anos_subjetivos_por_segundo:.4f} anos subjetivos"
    }


def main():
    print("=" * 70)
    print("  LAB01 — BENCHMARK DE FPS COMPUTACIONAL")
    print(f"  Hardware: {platform.processor() or platform.machine()}")
    print(f"  Python: {platform.python_version()}")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Duração por teste: {DURACAO_TESTE_S}s")
    print("=" * 70)
    print()

    resultados = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "hardware": {
                "processor": platform.processor() or platform.machine(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "os": f"{platform.system()} {platform.release()}"
            },
            "duracao_teste_s": DURACAO_TESTE_S,
            "seed": SEED
        },
        "benchmarks": {},
        "fps_biologicos": {},
        "dilatacao_cognitiva": {},
        "latencias_crompressor": {}
    }

    # ─── Benchmarks de Operações ─────────────────────────────
    testes = [
        ("int_add", "Adição Inteira (batched)", benchmark_aritmetica_int),
        ("float_mul", "Multiplicação Float (batched)", benchmark_aritmetica_float),
        ("sha256_512b", "SHA-256 (512 bytes)", benchmark_hash_sha256),
        ("xor_delta_4kb", "XOR Delta (4KB)", benchmark_xor_delta),
        ("mlp_forward_32x128x10", "MLP Forward (32→128→10)", benchmark_mlp_forward),
        ("cdc_rolling_1kb", "CDC Rolling Hash (1KB)", benchmark_cdc_simulado),
        ("codebook_lookup_k128", "Codebook Lookup (K=128, dim=16)", lambda d: benchmark_codebook_lookup(d, 128)),
        ("codebook_lookup_k256", "Codebook Lookup (K=256, dim=16)", lambda d: benchmark_codebook_lookup(d, 256)),
        ("codebook_lookup_k512", "Codebook Lookup (K=512, dim=16)", lambda d: benchmark_codebook_lookup(d, 512)),
        ("merkle_64_chunks", "Merkle Tree (64 chunks)", benchmark_merkle_hash),
    ]

    for key, nome, func in testes:
        print(f"  ▶ {nome}...", end="", flush=True)
        t_start = time.perf_counter()
        ops = func(DURACAO_TESTE_S)
        t_elapsed = time.perf_counter() - t_start
        ops_por_segundo = ops / t_elapsed

        resultados["benchmarks"][key] = {
            "nome": nome,
            "ops_total": ops,
            "tempo_real_s": round(t_elapsed, 4),
            "ops_por_segundo": round(ops_por_segundo, 2),
            "latencia_media_us": round(1_000_000 / ops_por_segundo, 4) if ops_por_segundo > 0 else None
        }
        print(f" {ops_por_segundo:,.0f} ops/s  ({1_000_000/ops_por_segundo:.2f} μs/op)")

    # ─── FPS Biológicos (dados de pesquisa) ─────────────────
    print()
    print("  ─── FPS Biológicos (Referências) ──────────────────")
    biologicos = {
        "humano": {"fps": FPS_HUMANO, "fonte": "Tobii — Speed of Visual Perception", "latencia_ms": "80-300ms"},
        "mosca": {"fps": FPS_MOSCA, "fonte": "Autrum, H. (1950) — Insect Vision", "latencia_ms": "~4ms"},
        "falcao_peregrino": {"fps": FPS_FALCAO, "fonte": "Potier et al. (2020) — CFF in raptors", "latencia_ms": "~8ms"},
        "cao": {"fps": FPS_CAO, "fonte": "Miller & Murphy (1995) — Canine CFF", "latencia_ms": "~12ms"},
        "polvo": {"fps": FPS_POLVO, "fonte": "Messenger (1981) — Cephalopod vision (estimativa)", "latencia_ms": "~33ms"},
    }
    for especie, dados in biologicos.items():
        print(f"    {especie:20s}: {dados['fps']:>5d} Hz  ({dados['latencia_ms']})")
    resultados["fps_biologicos"] = biologicos

    # ─── Dilatação Cognitiva ────────────────────────────────
    print()
    print("  ─── Dilatação Cognitiva (máquina vs biologia) ─────")

    # Usar o benchmark mais rápido como "FPS da máquina"
    fps_maquina_max = max(b["ops_por_segundo"] for b in resultados["benchmarks"].values())
    fps_maquina_tipico = resultados["benchmarks"]["sha256_512b"]["ops_por_segundo"]

    for especie, dados in biologicos.items():
        dil = calcular_dilatacao(fps_maquina_tipico, dados["fps"])
        resultados["dilatacao_cognitiva"][f"maquina_vs_{especie}"] = dil
        print(f"    vs {especie:15s}: ratio = {dil['ratio']:>15,.0f}x  |  {dil['descricao']}")

    # ─── Latências Crompressor ──────────────────────────────
    print()
    print("  ─── Latências Simuladas do Crompressor ────────────")

    crompressor_ops = ["xor_delta_4kb", "cdc_rolling_1kb", "codebook_lookup_k128",
                       "codebook_lookup_k256", "codebook_lookup_k512", "merkle_64_chunks",
                       "sha256_512b"]

    for key in crompressor_ops:
        b = resultados["benchmarks"][key]
        lat = b["latencia_media_us"]
        resultados["latencias_crompressor"][key] = {
            "latencia_us": lat,
            "ops_por_segundo": b["ops_por_segundo"]
        }
        print(f"    {b['nome']:40s}: {lat:>10.2f} μs/op  ({b['ops_por_segundo']:>12,.0f} ops/s)")

    # ─── Comparação com tempo humano ────────────────────────
    print()
    print("  ─── Tempo de 'pensamento' Crompressor vs Humano ───")
    TEMPO_RESPOSTA_HUMANO_MS = 300  # ms — resposta típica a estímulo visual

    for key in crompressor_ops:
        b = resultados["benchmarks"][key]
        lat_ms = b["latencia_media_us"] / 1000
        ratio = TEMPO_RESPOSTA_HUMANO_MS / lat_ms if lat_ms > 0 else float('inf')
        print(f"    {b['nome']:40s}: {ratio:>12,.0f}x mais rápido que humano")
        resultados["latencias_crompressor"][key]["ratio_vs_humano"] = round(ratio, 2)

    # ─── Salvar resultados ──────────────────────────────────
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTADOS_DIR, 'lab01_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    print()
    print(f"  ✅ Resultados salvos em: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
