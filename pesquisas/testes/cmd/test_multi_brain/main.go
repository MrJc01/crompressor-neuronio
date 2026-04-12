package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/MrJc01/crompressor-neuronio/pesquisas/testes/pkg/routing"
	"github.com/MrJc01/crompressor-neuronio/pesquisas/testes/pkg/tensor"
)

type BenchmarkResult struct {
	Brains          int     `json:"brains"`
	RoutingLatency  float64 `json:"routing_latency_us"`
	ComposeLatency  float64 `json:"compose_latency_us"`
	TotalLatency    float64 `json:"total_latency_us"`
	ThroughputMBps  float64 `json:"throughput_mbps"`
	MemoryOverheadMB float64 `json:"memory_overhead_mb"`
}

func main() {
	fmt.Println("╔═══════════════════════════════════════════════════════╗")
	fmt.Println("║   🧪 BENCHMARK - MULTI-BRAIN ROUTING (O(1))          ║")
	fmt.Println("╚═══════════════════════════════════════════════════════╝")

	chunkSize := 1024 * 1024 // 1 MB chunk (Standard FUSE/GGUF interception boundary estimate)
	originalChunk := make([]byte, chunkSize)

	// Mockup data for original chunk
	for i := range originalChunk {
		originalChunk[i] = 127
	}

	results := []BenchmarkResult{}

	// Test scaling from 1 to 5 brains
	for numBrains := 1; numBrains <= 5; numBrains++ {
		fmt.Printf("\n▶ Iniciando Benchmark com %d Neurônio(s)...\n", numBrains)
		
		router := routing.NewRouter()
		
		// Create mock context vector for query
		queryCtx := make([]float32, routing.ContextDim)
		queryCtx[0] = 1.0

		mockDeltas := make(map[string][]byte)

		// Register brains
		for i := 1; i <= numBrains; i++ {
			brainID := fmt.Sprintf("brain_%d", i)
			ctx := make([]float32, routing.ContextDim)
			ctx[i-1] = 1.0 // Orthogonal centroids for testing
			router.RegisterBrain(brainID, ctx)

			// Generate mock isolated delta tensor block
			d := make([]byte, chunkSize)
			for j := 0; j < chunkSize; j += 1024 {
				d[j] = byte(i * 10)
			}
			mockDeltas[brainID] = d
		}

		// 1. Benchmark: Routing (HNSW SIMD Simulation)
		startRouting := time.Now()
		weights, indices := router.GetTopKWeights(queryCtx, numBrains)
		routingElapsed := time.Since(startRouting).Microseconds()

		// Filter active deltas and weights based on Top-K output
		activeDeltas := make([][]byte, 0)
		activeWeights := make([]float32, 0)

		for i, idx := range indices {
			if weights[i] > 0.05 {
				brainID := router.Brains[idx].BrainID
				activeDeltas = append(activeDeltas, mockDeltas[brainID])
				activeWeights = append(activeWeights, weights[i])
			}
		}

		// 2. Benchmark: Composição em Memória
		startCompose := time.Now()
		composedChunk, err := tensor.ComposeDeltas(originalChunk, activeDeltas, activeWeights)
		if err != nil {
			log.Fatalf("Falha na composição: %v", err)
		}
		composeElapsed := time.Since(startCompose).Microseconds()

		// Memory estimate (naive): 1MB per delta in memory
		memoryOverhead := float64(numBrains * chunkSize) / (1024.0 * 1024.0)
		
		// Calculate Throughput on compose
		totalMicros := float64(composeElapsed)
		throughputMBps := 0.0
		if totalMicros > 0 {
			throughputMBps = float64(chunkSize) / (totalMicros / 1000000.0) / (1024 * 1024)
		}

		fmt.Printf("  └─ Roteamento HNSW (Top-%d): %d µs\n", numBrains, routingElapsed)
		fmt.Printf("  └─ Fusão Plástica (1 MB):    %d µs\n", composeElapsed)
		fmt.Printf("  └─ Throughput Aritmética:    %.2f MB/s\n", throughputMBps)
		fmt.Printf("  └─ Payload Hash Check:       %x...\n", composedChunk[:16])

		res := BenchmarkResult{
			Brains:          numBrains,
			RoutingLatency:  float64(routingElapsed),
			ComposeLatency:  float64(composeElapsed),
			TotalLatency:    float64(routingElapsed + composeElapsed),
			ThroughputMBps:  throughputMBps,
			MemoryOverheadMB: memoryOverhead,
		}
		results = append(results, res)
	}

	// Dump test results
	outputDir := filepath.Join("..", "dados")
	os.MkdirAll(outputDir, 0755)
	
	bytesJson, _ := json.MarshalIndent(results, "", "  ")
	outputPath := filepath.Join(outputDir, "fase3_routing.json")
	
	err := ioutil.WriteFile(outputPath, bytesJson, 0644)
	if err != nil {
		log.Printf("Aviso: Falha ao escrever %s: %v", outputPath, err)
	} else {
		fmt.Printf("\n✅ Relatório de engenharia extraído com sucesso para: %s\n", outputPath)
	}
}
