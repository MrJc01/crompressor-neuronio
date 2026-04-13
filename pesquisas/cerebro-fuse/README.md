# 🧠 Cérebro-FUSE: Orquestração Multi-Brain via FUSE

> **Linha de Pesquisa:** Tratar LLMs inteiros como neurônios congelados (.crom) e orquestrá-los via FUSE kernel driver com deltas XOR compostos.

---

## O Que É

Esta linha de pesquisa explora a ideia de **congelar modelos LLM completos** no formato `.crom` (DNA Base-4 + Codebook + Merkle Tree) e servi-los via **FUSE filesystem**, permitindo:

- Leitura O(1) do SSD sem carregar o modelo inteiro na RAM
- Aplicação de tensores delta XOR em tempo real no kernel (interceptor de Read)
- Roteamento dinâmico entre múltiplos "cérebros" congelados via similaridade de cosseno
- Composição ponderada de deltas de múltiplos cérebros para "criatividade emergente"

## Componentes

| Componente | Descrição | Status |
|---|---|---|
| `testes/cmd/fuse_test/` | Motor principal: FUSE + API + LLM | Funcional (PoC) |
| `testes/cmd/gguf_to_crom/` | Parser GGUF → .crom via FastCDC | Funcional (CDC real) |
| `testes/pkg/fuse_mount/` | Driver FUSE com composição multi-delta | Funcional |
| `testes/pkg/routing/` | Roteador por similaridade de cosseno | Funcional |
| `testes/pkg/api/` | API REST para o Web Cockpit | Funcional |
| `testes/pkg/engine/` | Cliente LLM (llama.cpp) | Funcional |
| `web-cockpit/` | UI React com Canvas pipeline graph | Funcional |

## Tese Central

> "Um modelo inteiro congelado pode servir como neurônio fixo, e deltas XOR esparsos são suficientes para gerar saídas adaptativas sem retraining."

## O Que Falta Provar

1. Pipeline real GGUF → `.crom` binário (não JSON)
2. FUSE lendo do `.crom` sem carregar tudo na RAM
3. Benchmark: leitura FUSE .crom vs leitura direta .gguf
4. Evidência: deltas XOR sobre chunks reais produzem output interpretável

---

*Código principal em: `pesquisas/testes/` e `web-cockpit/`*
