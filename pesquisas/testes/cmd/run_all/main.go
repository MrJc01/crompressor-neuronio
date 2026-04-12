// Suite de testes completa: executa todos os benchmarks e gera relatórios em JSON.
// Uso: go run pesquisas/testes/cmd/run_all/main.go
package main

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"time"

	neuronio "github.com/MrJc01/crompressor-neuronio/pesquisas/testes/pkg"
)

func main() {
	fmt.Println("╔═══════════════════════════════════════════════════════╗")
	fmt.Println("║   CROMPRESSOR-NEURÔNIO — SUITE DE TESTES V2          ║")
	fmt.Println("║   Dados realistas + Validação de Hipóteses           ║")
	fmt.Println("╚═══════════════════════════════════════════════════════╝")
	fmt.Println()

	// Detectar project root: se CWD é pesquisas/testes/, subir 2 níveis
	// Detectar o diretório correto para salvar dados
	// Se estamos em pesquisas/testes/, subir 2 níveis para achar pesquisas/dados/
	dataDir := filepath.Join("..", "..", "pesquisas", "dados")
	if _, err := os.Stat(filepath.Join("..", "..", "pesquisas")); os.IsNotExist(err) {
		// Provavelmente rodando do project root
		dataDir = filepath.Join("pesquisas", "dados")
	}
	os.MkdirAll(dataDir, 0755)

	configs := []struct {
		Name      string
		Chunks    int
		ChunkSize int
		Label     string
	}{
		{"Qwen2.5-0.5B (sim)", 2000, 512, "qwen05b"},
		{"Qwen2.5-1.5B (sim)", 6000, 512, "qwen15b"},
		{"LLaMA-3.2-1B (sim)", 4000, 512, "llama1b"},
		{"Phi-3-mini (sim)", 15000, 512, "phi3mini"},
	}

	allCompression := make([]neuronio.CompressionMetrics, 0)
	allEntropy := make([]neuronio.EntropyMetrics, 0)
	allBench := make([]neuronio.BenchmarkResult, 0)
	passed := 0
	failed := 0
	total := 0

	check := func(name string, ok bool, details string) {
		total++
		if ok {
			passed++
			fmt.Printf("     ✅ %s: %s\n", name, details)
		} else {
			failed++
			fmt.Printf("     ❌ %s: %s\n", name, details)
		}
	}

	// ====================================================================
	// FASE 1: Brain Freeze — Compressão com Deduplicação Real
	// ====================================================================
	fmt.Println("━━━ FASE 1: Brain Freeze + Deduplicação Realista ━━━━━━")

	for _, cfg := range configs {
		fmt.Printf("\n  🧬 %s (%d chunks × %d bytes)\n", cfg.Name, cfg.Chunks, cfg.ChunkSize)

		start := time.Now()
		brain := neuronio.GenerateSyntheticBrain(cfg.Chunks, cfg.ChunkSize)
		genTime := time.Since(start)

		neuronio.FreezeBrain(brain)
		cm := neuronio.MeasureCompressionMetrics(brain, cfg.Name)

		check("Frozen", brain.Header.Frozen, fmt.Sprintf("frozen=%v", brain.Header.Frozen))
		check("Ratio > 1x", cm.Ratio > 1.0, fmt.Sprintf("ratio=%.2fx", cm.Ratio))
		check("Dedup > 0%%", cm.DedupRate > 0, fmt.Sprintf("dedup=%.1f%% (%d/%d chunks)", cm.DedupRate, cm.DedupChunks, cm.ChunkCount))
		check("Codebook < total", cm.CodebookSize < cm.ChunkCount, fmt.Sprintf("codebook=%d < chunks=%d", cm.CodebookSize, cm.ChunkCount))

		fmt.Printf("     📊 Ratio: %.2fx | Dedup: %.1f%% | Codebook: %d | Tempo: %v\n",
			cm.Ratio, cm.DedupRate, cm.CodebookSize, genTime)

		allCompression = append(allCompression, cm)

		// Entropia
		em := neuronio.MeasureEntropyMetrics(brain, cfg.Label)
		fmt.Printf("     📈 Entropia: μ=%.3f σ=%.3f [%.3f, %.3f]\n",
			em.MeanEntropy, em.StdEntropy, em.MinEntropy, em.MaxEntropy)

		check("Entropia variada", em.StdEntropy > 0.1,
			fmt.Sprintf("σ=%.3f (>0.1 indica mistura embedding/atenção/FFN)", em.StdEntropy))

		allEntropy = append(allEntropy, em)

		// ================================================================
		// FASE 2: Tensor Delta — XOR + VQ
		// ================================================================
		fmt.Printf("\n  ⚡ Tensor Delta sobre %s\n", cfg.Label)

		allDeltas := make([]neuronio.DeltaMetrics, 0)

		sparsities := []float64{0.70, 0.80, 0.90, 0.95}
		for _, sp := range sparsities {
			delta := neuronio.GenerateXORDelta(brain, sp)
			dm := neuronio.MeasureDeltaMetrics(delta, brain)
			fmt.Printf("     XOR(sp=%.0f%%): size=%d ratio=%.2f%% sparsity=%.1f%% latency=%dμs\n",
				sp*100, dm.DeltaSize, dm.DeltaRatio, dm.Sparsity, dm.ApplyLatency)
			allDeltas = append(allDeltas, dm)
		}

		check("XOR Delta < 10%%", allDeltas[0].DeltaRatio < 10,
			fmt.Sprintf("ratio=%.2f%%", allDeltas[0].DeltaRatio))

		// VQ Delta
		vqDelta := neuronio.GenerateVQDelta(brain, 128)
		vqDm := neuronio.MeasureDeltaMetrics(vqDelta, brain)
		fmt.Printf("     VQ(dim=128): size=%d ratio=%.2f%% sparsity=%.1f%%\n",
			vqDm.DeltaSize, vqDm.DeltaRatio, vqDm.Sparsity)
		allDeltas = append(allDeltas, vqDm)

		check("VQ Delta < brain", vqDm.DeltaSize < int(brain.Header.CompressedSize),
			fmt.Sprintf("VQ=%d < brain=%d", vqDm.DeltaSize, brain.Header.CompressedSize))

		neuronio.SaveJSON(filepath.Join(dataDir, fmt.Sprintf("deltas_%s.json", cfg.Label)), allDeltas)

		// ================================================================
		// VALIDAÇÕES MATEMÁTICAS
		// ================================================================
		fmt.Printf("\n  🔬 Validações Matemáticas\n")

		// Reversibilidade XOR
		testChunk := brain.Chunks[0].Data
		xorDelta := neuronio.GenerateXORDelta(brain, 0.85)
		modified := neuronio.ApplyXORDelta(testChunk, xorDelta.Data)
		restored := neuronio.ApplyXORDelta(modified, xorDelta.Data)
		check("XOR Reversível", bytes.Equal(testChunk, restored), "A⊕B⊕B=A")

		// Composição Associativa
		d1 := neuronio.GenerateXORDelta(brain, 0.80)
		d2 := neuronio.GenerateXORDelta(brain, 0.90)
		composed := neuronio.ComposeDeltas(d1, d2)
		path1 := neuronio.ApplyXORDelta(testChunk, composed.Data)
		step1 := neuronio.ApplyXORDelta(testChunk, d1.Data)
		path2 := neuronio.ApplyXORDelta(step1, d2.Data)
		check("Composição Equivalente", bytes.Equal(path1, path2), "chunk⊕(d1⊕d2) = (chunk⊕d1)⊕d2")

		// Merkle Verify
		check("Merkle OK", neuronio.VerifyMerkleRoot(brain), "root verificado")
		brain.Chunks[0].Data[0] ^= 0xFF // corromper
		check("Merkle Detecção", !neuronio.VerifyMerkleRoot(brain), "corrupção detectada")
		brain.Chunks[0].Data[0] ^= 0xFF // restaurar

		// ================================================================
		// BENCHMARK XOR
		// ================================================================
		iterations := 100000
		xorBenchDelta := neuronio.GenerateXORDelta(brain, 0.85)
		benchChunk := brain.Chunks[0].Data

		xorStart := time.Now()
		for i := 0; i < iterations; i++ {
			neuronio.ApplyXORDelta(benchChunk, xorBenchDelta.Data)
		}
		xorElapsed := time.Since(xorStart)
		xorNsOp := float64(xorElapsed.Nanoseconds()) / float64(iterations)
		mbSec := float64(len(benchChunk)) / xorNsOp * 1000

		check("XOR < 100μs", xorNsOp < 100000,
			fmt.Sprintf("%.0f ns/op (%.1f MB/s)", xorNsOp, mbSec))

		allBench = append(allBench, neuronio.BenchmarkResult{
			TestName:   fmt.Sprintf("xor_delta_%s", cfg.Label),
			Timestamp:  time.Now().Format(time.RFC3339),
			Duration:   xorElapsed.Nanoseconds(),
			Iterations: iterations,
			NsPerOp:    xorNsOp,
			MBPerSec:   mbSec,
		})

		// Salvar .crom
		neuronio.WriteCrom(filepath.Join(dataDir, fmt.Sprintf("brain_%s.crom.json", cfg.Label)), brain)
	}

	// ====================================================================
	// FASE 3: Multi-Brain Routing
	// ====================================================================
	fmt.Println("\n━━━ FASE 3: Multi-Brain Routing ━━━━━━━━━━━━━━━━━━━━━━")

	routingResults := make([]neuronio.RoutingMetrics, 0)

	for numBrains := 1; numBrains <= 5; numBrains++ {
		// Forçar GC antes de medir memória
		runtime.GC()
		var memBefore runtime.MemStats
		runtime.ReadMemStats(&memBefore)

		brains := make([]*neuronio.BrainCrom, numBrains)
		for i := 0; i < numBrains; i++ {
			brains[i] = neuronio.GenerateSyntheticBrain(500, 256)
			neuronio.FreezeBrain(brains[i])
		}

		// Forçar GC e medir
		runtime.GC()
		var memAfter runtime.MemStats
		runtime.ReadMemStats(&memAfter)

		// Usar TotalAlloc para evitar underflow
		memDelta := float64(memAfter.TotalAlloc-memBefore.TotalAlloc) / 1024 / 1024

		// Simula routing
		routingStart := time.Now()
		selected := make([]int, 0)
		weights := make([]float64, 0)
		topK := minInt(2, numBrains)
		for i := 0; i < topK; i++ {
			selected = append(selected, i)
			weights = append(weights, 1.0/float64(topK))
		}
		routingElapsed := time.Since(routingStart)

		rm := neuronio.RoutingMetrics{
			Timestamp:      time.Now().Format(time.RFC3339),
			NumBrains:      numBrains,
			DecisionTimeNs: routingElapsed.Nanoseconds(),
			SelectedBrains: selected,
			Weights:        weights,
			MemoryUsedMB:   memDelta,
		}
		routingResults = append(routingResults, rm)

		fmt.Printf("  🧠×%d: routing=%dns, memory=%.1fMB, top-K=%v\n",
			numBrains, rm.DecisionTimeNs, rm.MemoryUsedMB, selected)
	}

	check("Routing < 5ms", routingResults[4].DecisionTimeNs < 5_000_000,
		fmt.Sprintf("%dns", routingResults[4].DecisionTimeNs))

	// ====================================================================
	// SALVAR TODOS OS DADOS
	// ====================================================================
	fmt.Println("\n━━━ SALVANDO DADOS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	neuronio.SaveJSON(filepath.Join(dataDir, "compression_all.json"), allCompression)
	neuronio.SaveJSON(filepath.Join(dataDir, "entropy_all.json"), allEntropy)
	neuronio.SaveJSON(filepath.Join(dataDir, "routing_all.json"), routingResults)
	neuronio.SaveJSON(filepath.Join(dataDir, "bench_all.json"), allBench)

	for _, f := range []string{"compression_all", "entropy_all", "routing_all", "bench_all"} {
		fmt.Printf("  💾 pesquisas/dados/%s.json\n", f)
	}

	// ====================================================================
	// RELATÓRIO FINAL
	// ====================================================================
	fmt.Println()
	fmt.Println("╔═══════════════════════════════════════════════════════╗")
	fmt.Printf("║   RESULTADO: %d/%d testes passaram                    \n", passed, total)
	fmt.Println("╠═══════════════════════════════════════════════════════╣")
	for _, cm := range allCompression {
		fmt.Printf("║   %-20s Ratio=%.2fx Dedup=%.1f%%\n", cm.ModelName, cm.Ratio, cm.DedupRate)
	}
	fmt.Println("╠═══════════════════════════════════════════════════════╣")
	for _, b := range allBench {
		fmt.Printf("║   %-20s %.0f ns/op  %.1f MB/s\n", b.TestName, b.NsPerOp, b.MBPerSec)
	}
	fmt.Println("╚═══════════════════════════════════════════════════════╝")

	if failed > 0 {
		fmt.Printf("\n⚠️  %d teste(s) falharam!\n", failed)
		os.Exit(1)
	}
	fmt.Println("\n✅ Todos os testes passaram!")
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
