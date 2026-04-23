#!/usr/bin/env python3
"""
CROM Ecosystem — Runner Unificado
==================================
Executa todos os testes em sequência e exporta um .jsonl com métricas auditáveis.

Uso: .venv/bin/python run_all.py
"""
import subprocess
import sys
import time
import json
import os

VENV_PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TESTS = [
    {
        "id": "02_p2p",
        "name": "P2P Ed25519 Real",
        "script": "02_rede_p2p/p2p_network.py",
    },
    {
        "id": "03_compressor",
        "name": "Compressor VQ (Pesos GPT-2 Reais)",
        "script": "03_compressor_cli/compressor.py",
    },
    {
        "id": "05_mapa",
        "name": "Pathfinder Mapa Real (OpenStreetMap)",
        "script": "05_mapa_real/real_map_pathfinder.py",
    },
    # Chat é separado por ser pesado (~3 min na CPU)
    # Rodar manualmente: .venv/bin/python 01_chat_blindado/crom_chat.py
]

class Colors:
    HEADER = '\033[95m'
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def run_test(test_info):
    """Executa um teste e captura stdout/stderr."""
    script = os.path.join(BASE_DIR, test_info["script"])
    
    print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"  ▶ {test_info['name']}")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}\n")
    
    t0 = time.perf_counter()
    
    try:
        result = subprocess.run(
            [VENV_PYTHON, script],
            capture_output=False,
            text=True,
            timeout=120,
            cwd=BASE_DIR,
        )
        elapsed = time.perf_counter() - t0
        success = result.returncode == 0
        
        if success:
            print(f"\n{Colors.OKGREEN}  ✅ PASSOU ({elapsed:.1f}s){Colors.ENDC}")
        else:
            print(f"\n{Colors.FAIL}  ❌ FALHOU (exit code {result.returncode}){Colors.ENDC}")
        
        return {
            "test_id": test_info["id"],
            "name": test_info["name"],
            "status": "PASS" if success else "FAIL",
            "elapsed_s": round(elapsed, 2),
            "exit_code": result.returncode,
            "timestamp": time.time(),
        }
        
    except subprocess.TimeoutExpired:
        print(f"\n{Colors.FAIL}  ⏰ TIMEOUT (>120s){Colors.ENDC}")
        return {
            "test_id": test_info["id"],
            "name": test_info["name"],
            "status": "TIMEOUT",
            "elapsed_s": 120,
            "timestamp": time.time(),
        }


def main():
    print(f"{Colors.BOLD}")
    print(f"╔══════════════════════════════════════════════════════════════════════════════╗")
    print(f"║  CROM Neural Ecosystem — Suite de Validação (Dados Reais)                  ║")
    print(f"╚══════════════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")
    
    print(f"[SYS] Executando {len(TESTS)} testes em sequência...")
    print(f"[SYS] O teste do Chat (01) é pesado (~3min CPU) e deve ser rodado separadamente.")
    
    results = []
    for test in TESTS:
        result = run_test(test)
        results.append(result)
    
    # Exportar JSONL
    log_file = os.path.join(BASE_DIR, "crom_audit.jsonl")
    with open(log_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    # Resumo
    passed = sum(1 for r in results if r["status"] == "PASS")
    total = len(results)
    total_time = sum(r["elapsed_s"] for r in results)
    
    print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}  RESUMO: {passed}/{total} testes passaram | Tempo total: {total_time:.1f}s{Colors.ENDC}")
    print(f"  Logs exportados para: {log_file}")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
    
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
