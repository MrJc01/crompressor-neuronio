# 🚀 Eixo 05 — Deployment: Edge, WASM e Publicação

> **Pergunta Central:** Como empacotar o motor CROM em <10MB para rodar no browser sem instalação?

---

## Contexto

O Go compila para WASM via TinyGo. WasmEdge (CNCF) suporta WASI-NN para aceleração de hardware. Em 2026, WASM é a plataforma padrão para edge AI.

## Documentos Neste Eixo

| Arquivo | Foco |
|:--------|:-----|
| [wasm-deployment.md](wasm-deployment.md) | TinyGo → WASM → browser |
| [arm-deployment.md](arm-deployment.md) | Cross-compilation para ARM (Raspberry Pi, mobile) |
| [paper-arxiv.md](paper-arxiv.md) | Estrutura do paper para publicação |

## Tese Central

> O motor CROM compilado para WASM executa no browser com **zero dependência, zero instalação, zero cloud**. Inferência soberana: os dados nunca saem do dispositivo do usuário. Ed25519 garante integridade criptográfica.

## Tecnologias

| Runtime | Cold Start | Memória | Ideal para |
|---------|-----------|---------|------------|
| WasmEdge | ~1.5ms | ~8MB | LLM inference + plugins |
| Wasmtime | ~2ms | ~12MB | Security-first |
| Wasmer | ~1ms | ~10MB | Performance geral |
| TinyGo target | - | <5MB | Nossa compilação |
