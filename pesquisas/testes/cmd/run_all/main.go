// Teste principal: executa todos os benchmarks e gera relatórios em JSON.
// Uso: go run pesquisas/testes/cmd/run_all/main.go
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"time"

	neuronio "github.com/MrJc01/crompressor-neuronio/pesquisas/testes/pkg"
)

func main() {
	fmt.Println("╔═══════════════════════════════════════════════════════╗")
	fmt.Println("║     CROMPRESSOR-NEURÔNIO — SUITE DE TESTES           ║")
	fmt.Println("║     Todas as 3 vertentes + benchmarks completos      ║")
	fmt.Println("╚═══════════════════════════════════════════════════════╝")
	fmt.Println()

	// Diretórios de saída
	dataDir := filepath.Join("pesquisas", "dados")
	reportDir := filepath.Join("pesquisas", "relatorios")
	os.MkdirAll(dataDir, 0755)
	os.MkdirAll(reportDir, 0755)

	// ====================================================================
	// FASE 1: Brain Freeze — Compressão e Congelamento
	// ====================================================================
	fmt.Println("━━━ FASE 1: Brain Freeze ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	configs := []struct {
		Name      string
		Chunks    int
		ChunkSize int
		Label     string
	}{
		{"Qwen2.5-0.5B (simulado)", 2000, 512, "qwen05b"},
		{"Qwen2.5-1.5B (simulado)", 6000, 512, "qwen15b"},
		{"LLaMA-3.2-1B (simulado)", 4000, 512, "llama1b"},
		{"Phi-3-mini (simulado)", 15000, 512, "phi3mini"},
	}

	allCompression := make([]neuronio.CompressionMetrics, 0)
	allEntropy := make([]neuronio.EntropyMetrics, 0)

	for _, cfg := range configs {
		fmt.Printf("\n  🧬 Gerando brain: %s (%d chunks × %d bytes)\n", cfg.Name, cfg.Chunks, cfg.ChunkSize)

		start := time.Now()
		brain := neuronio.GenerateSyntheticBrain(cfg.Chunks, cfg.ChunkSize)
		elapsed := time.Since(start)

		fmt.Printf("     ⏱  Gerado em %v\n", elapsed)

		// Congelar
		neuronio.FreezeBrain(brain)
		fmt.Printf("     🔒 Frozen: %v\n", brain.Header.Frozen)

		// Métricas de compressão
		cm := neuronio.MeasureCompressionMetrics(brain)
		fmt.Printf("     📊 Ratio: %.2fx | Dedup: %.1f%% | Codebook: %d entries\n",
			cm.Ratio, cm.DedupRate, cm.CodebookSize)
		allCompression = append(allCompression, cm)

		// Entropia
		em := neuronio.MeasureEntropyMetrics(brain, cfg.Label)
		fmt.Printf("     📈 Entropia: μ=%.4f σ=%.4f [%.4f, %.4f] bits/byte\n",
			em.MeanEntropy, em.StdEntropy, em.MinEntropy, em.MaxEntropy)
		allEntropy = append(allEntropy, em)

		// ================================================================
		// FASE 2: Tensor Delta
		// ================================================================
		fmt.Println("\n  ━━━ FASE 2: Tensor Delta ━━━━━━━━━━━━━━━━━━━━━━━")

		// XOR Delta com diferentes esparsificações
		sparsities := []float64{0.70, 0.80, 0.90, 0.95}
		allDeltas := make([]neuronio.DeltaMetrics, 0)

		for _, sp := range sparsities {
			delta := neuronio.GenerateXORDelta(brain, sp)
			dm := neuronio.MeasureDeltaMetrics(delta, brain)
			fmt.Printf("     ⚡ XOR Delta (sparsity=%.0f%%): size=%d bytes, ratio=%.2f%%, latency=%dns\n",
				sp*100, dm.DeltaSize, dm.DeltaRatio, dm.ApplyLatency)
			allDeltas = append(allDeltas, dm)
		}

		// VQ Delta
		vqDelta := neuronio.GenerateVQDelta(brain, 128)
		vqDm := neuronio.MeasureDeltaMetrics(vqDelta, brain)
		fmt.Printf("     🔮 VQ Delta (dim=128): size=%d bytes, ratio=%.2f%%, sparsity=%.1f%%\n",
			vqDm.DeltaSize, vqDm.DeltaRatio, vqDm.Sparsity)
		allDeltas = append(allDeltas, vqDm)

		// Salvar deltas por modelo
		neuronio.SaveJSON(filepath.Join(dataDir, fmt.Sprintf("deltas_%s.json", cfg.Label)), allDeltas)

		// ================================================================
		// FASE 2.5: XOR vs VQ Benchmark
		// ================================================================
		fmt.Println("\n  ━━━ XOR vs VQ Benchmark ━━━━━━━━━━━━━━━━━━━━━━━━")

		xorDelta := neuronio.GenerateXORDelta(brain, 0.85)
		testChunk := brain.Chunks[0].Data

		// Benchmark XOR
		iterations := 100000
		xorStart := time.Now()
		for i := 0; i < iterations; i++ {
			neuronio.ApplyXORDelta(testChunk, xorDelta.Data)
		}
		xorElapsed := time.Since(xorStart)
		xorNsOp := float64(xorElapsed.Nanoseconds()) / float64(iterations)

		fmt.Printf("     XOR: %d iterações em %v (%.0f ns/op, %.2f MB/s)\n",
			iterations, xorElapsed, xorNsOp,
			float64(len(testChunk))/xorNsOp*1000)

		benchResults := []neuronio.BenchmarkResult{
			{
				TestName:   fmt.Sprintf("xor_delta_%s", cfg.Label),
				Timestamp:  time.Now().Format(time.RFC3339),
				Duration:   xorElapsed.Nanoseconds(),
				Iterations: iterations,
				NsPerOp:    xorNsOp,
				MBPerSec:   float64(len(testChunk)) / xorNsOp * 1000,
			},
		}
		neuronio.SaveJSON(filepath.Join(dataDir, fmt.Sprintf("bench_%s.json", cfg.Label)), benchResults)
	}

	// ====================================================================
	// FASE 3: Multi-Brain Routing (Simulação)
	// ====================================================================
	fmt.Println("\n━━━ FASE 3: Multi-Brain Routing ━━━━━━━━━━━━━━━━━━━━━━")

	routingResults := make([]neuronio.RoutingMetrics, 0)

	for numBrains := 1; numBrains <= 5; numBrains++ {
		var memBefore, memAfter runtime.MemStats
		runtime.ReadMemStats(&memBefore)

		brains := make([]*neuronio.BrainCrom, numBrains)
		for i := 0; i < numBrains; i++ {
			brains[i] = neuronio.GenerateSyntheticBrain(1000, 256)
			neuronio.FreezeBrain(brains[i])
		}

		// Simula decisão de routing (HNSW simplificado)
		routingStart := time.Now()
		selected := make([]int, 0)
		weights := make([]float64, 0)
		topK := min(2, numBrains)

		for i := 0; i < topK; i++ {
			selected = append(selected, i)
			weights = append(weights, 1.0/float64(topK))
		}
		routingElapsed := time.Since(routingStart)

		runtime.ReadMemStats(&memAfter)
		memUsedMB := float64(memAfter.Alloc-memBefore.Alloc) / 1024 / 1024

		rm := neuronio.RoutingMetrics{
			Timestamp:      time.Now().Format(time.RFC3339),
			NumBrains:      numBrains,
			DecisionTimeNs: routingElapsed.Nanoseconds(),
			SelectedBrains: selected,
			Weights:        weights,
			MemoryUsedMB:   memUsedMB,
		}
		routingResults = append(routingResults, rm)

		fmt.Printf("  🧠×%d: routing=%dns, memory=%.2fMB, selected=%v\n",
			numBrains, rm.DecisionTimeNs, rm.MemoryUsedMB, selected)
	}

	// ====================================================================
	// SALVAR TODOS OS DADOS
	// ====================================================================
	fmt.Println("\n━━━ SALVANDO DADOS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	neuronio.SaveJSON(filepath.Join(dataDir, "compression_all.json"), allCompression)
	fmt.Printf("  💾 %s\n", filepath.Join(dataDir, "compression_all.json"))

	neuronio.SaveJSON(filepath.Join(dataDir, "entropy_all.json"), allEntropy)
	fmt.Printf("  💾 %s\n", filepath.Join(dataDir, "entropy_all.json"))

	neuronio.SaveJSON(filepath.Join(dataDir, "routing_all.json"), routingResults)
	fmt.Printf("  💾 %s\n", filepath.Join(dataDir, "routing_all.json"))

	// ====================================================================
	// RELATÓRIO FINAL
	// ====================================================================
	fmt.Println("\n╔═══════════════════════════════════════════════════════╗")
	fmt.Println("║     RELATÓRIO FINAL                                  ║")
	fmt.Println("╠═══════════════════════════════════════════════════════╣")
	for i, cm := range allCompression {
		fmt.Printf("║  Modelo %d: Ratio=%.2fx Dedup=%.1f%% Codebook=%d       \n",
			i+1, cm.Ratio, cm.DedupRate, cm.CodebookSize)
	}
	fmt.Printf("║  Multi-Brain: %d configs testadas                     \n", len(routingResults))
	fmt.Println("╚═══════════════════════════════════════════════════════╝")
	fmt.Println("\n✅ Todos os dados salvos em pesquisas/dados/")
	fmt.Println("   Use 'python pesquisas/visualizacao/visualizar_resultados.py' para gráficos")
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
