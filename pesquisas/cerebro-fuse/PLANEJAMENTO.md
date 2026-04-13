# 📋 Cérebro-FUSE — Planejamento & Checklist

> **Status:** ⏸️ Pausado — Priorizando pesquisa Tensor-Vivo
> **Última atualização:** 2026-04-13

---

## Diagnóstico (Análise Completa Feita)

### O Que Está Bom
- ✅ Tese original e com base acadêmica
- ✅ `neuronio.go` matematicamente correto (Shannon, XOR, Merkle, DNA roundtrip)
- ✅ 17 testes unitários + 5 benchmarks passando
- ✅ FUSE driver funcional com go-fuse/v2 real
- ✅ `gguf_parser` **já usa FastCDC real** sobre o GGUF (não é simulação)
- ✅ Web Cockpit UI bonita com Canvas render
- ✅ LLM Client conecta ao llama.cpp real

### O Que Precisa Melhorar
- ❌ Maioria dos dados é simulação sintética (GenerateSyntheticBrain)
- ❌ FUSE carrega modelo inteiro na RAM (invalida tese O(1) SSD)
- ❌ Formato .crom é JSON, não binário
- ❌ "HNSW" é brute-force cosine (nomenclatura enganosa)
- ❌ VQ Delta é fake (bytes aleatórios, não k-means)
- ❌ Zero integração com crompressor core
- ❌ Race condition em ActiveContext (escrita sem lock)
- ❌ APIs deprecated (ioutil)

---

## Fases Planejadas

### Fase 0: Limpeza Técnica (~1h)
- [ ] Substituir `ioutil.ReadFile` → `os.ReadFile` em `fuse_test/main.go`
- [ ] Substituir `ioutil.ReadAll` → `io.ReadAll` em `llm.go` (linhas 42, 94)
- [ ] Substituir `ioutil.WriteFile` → `os.WriteFile` em `test_multi_brain/main.go`
- [ ] Remover imports de `io/ioutil`
- [ ] Proteger `ActiveContext` com lock em `rest.go` e `crom_fs.go`
- [ ] Adicionar error handling em `rand.Read()` (neuronio.go)
- [ ] `go vet ./...` limpo
- [ ] `go build -race ./...` sem warnings

### Fase 1: Pipeline Real GGUF → .crom Binário (~4-6h)
- [ ] Criar `pkg/crom/format.go` — header binário 64 bytes + ChunkIndex
- [ ] Criar `pkg/crom/writer.go` — serialização binária
- [ ] Criar `pkg/crom/reader.go` — leitura com acesso aleatório
- [ ] Criar `pkg/crom/format_test.go` — roundtrip, magic, Merkle
- [ ] Renomear `gguf_parser` → `gguf_to_crom`, saída binária
- [ ] Gerar `brain_qwen.crom` binário real
- [ ] Criar `cmd/gen_delta/main.go` — delta XOR real
- [ ] Testar reversibilidade com dados reais

### Fase 2: FUSE Lendo de .crom Real (~4-6h)
- [ ] Refatorar `CromNode` para leitura lazy (sem carregar tudo na RAM)
- [ ] `cromReader *crom.Reader` em vez de `frozenData []byte`
- [ ] `Read()` calcula quais chunks cobrem o range pedido
- [ ] Fallback graceful para .gguf direto se .crom não existir
- [ ] Criar `cmd/bench_fuse/main.go` — benchmark FUSE vs leitura direta
- [ ] Verificar overhead FUSE < 10%

### Fase 3: Honestidade Arquitetural (~2h)
- [ ] Renomear `HNSWRouter` → `CosineRouter`
- [ ] Deletar `GenerateVQDelta()` fake
- [ ] Atualizar docs para refletir realidade
- [ ] Adicionar `// STUB:` nos simuladores restantes
- [ ] `README.md` com seção "Status Real"

### Fase 4: Cockpit UI com Dados Reais (~3-4h)
- [ ] Endpoint `/crominfo` com dados do .crom real
- [ ] Painel de métricas real na UI
- [ ] Heatmap com chunks reais
- [ ] Labels honestos (Cosine Router, modo legado)

### Fase 5: Validação Empírica (~2-3h)
- [ ] `cmd/proof_of_concept/main.go` — pipeline end-to-end
- [ ] Relatório `prova_minima.md` com dados mensuráveis
- [ ] Suite de testes final: `go test`, `go vet`, `go build -race`

---

## Estimativa Total: 16-22h (~3-4 dias focados)

## Referências
- Análise completa: conversa `ed878ebf` (13/04/2026)
- Código principal: `pesquisas/testes/` e `web-cockpit/`
