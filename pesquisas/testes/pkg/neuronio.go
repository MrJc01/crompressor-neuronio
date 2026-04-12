// Package neuronio fornece primitivas de teste para as 3 vertentes do crompressor-neuronio.
// Simula operações de XOR Delta, Vector Quantization, entropia de Shannon e
// composição multi-brain, gerando dados mensuráveis em JSON para análise.
package neuronio

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"time"
)

// ============================================================================
// TIPOS FUNDAMENTAIS
// ============================================================================

// BrainCrom representa um "cérebro" congelado (neurônio fixo).
// Simula o formato .crom com chunks, codebook e merkle.
type BrainCrom struct {
	Header   CromHeader   `json:"header"`
	Chunks   []Chunk      `json:"chunks"`
	Codebook []CodeEntry  `json:"codebook"`
	Merkle   MerkleTree   `json:"merkle"`
}

// CromHeader contém metadados do arquivo .crom
type CromHeader struct {
	Magic          string `json:"magic"`           // "CROM"
	Version        int    `json:"version"`         // 1
	Frozen         bool   `json:"frozen"`          // true se congelado
	Domain         string `json:"domain"`          // "brain"
	ChunkCount     int    `json:"chunk_count"`
	CodebookSize   int    `json:"codebook_size"`
	OriginalSize   int64  `json:"original_size"`
	CompressedSize int64  `json:"compressed_size"`
	MerkleRoot     string `json:"merkle_root"`
}

// Chunk representa um bloco CDC de dados
type Chunk struct {
	ID       int    `json:"id"`
	Hash     string `json:"hash"`
	Data     []byte `json:"-"`
	Size     int    `json:"size"`
	Entropy  float64 `json:"entropy"`
	IsDelta  bool   `json:"is_delta"`   // true se é referência XOR
	DeltaRef int    `json:"delta_ref"`  // ID do chunk referenciado
}

// CodeEntry representa uma entrada no codebook DNA
type CodeEntry struct {
	Hash     uint64 `json:"hash"`
	DNA      string `json:"dna"`       // sequência A/T/C/G
	RefCount int    `json:"ref_count"` // quantas vezes referenciado
}

// MerkleTree representa a árvore de integridade
type MerkleTree struct {
	Root   string   `json:"root"`
	Leaves []string `json:"leaves"`
	Depth  int      `json:"depth"`
}

// TensorDelta representa um tensor delta para aplicar sobre um brain.crom
type TensorDelta struct {
	Type         string  `json:"type"`          // "xor", "vq", "composed"
	TargetHash   string  `json:"target_hash"`   // hash do brain alvo
	Data         []byte  `json:"-"`
	Size         int     `json:"size"`
	NonZeroRatio float64 `json:"non_zero_ratio"`
	Sparsity     float64 `json:"sparsity"`
}

// ============================================================================
// MÉTRICAS E RELATÓRIOS
// ============================================================================

// CompressionMetrics contém métricas de compressão
type CompressionMetrics struct {
	Timestamp       string  `json:"timestamp"`
	OriginalSize    int64   `json:"original_size_bytes"`
	CompressedSize  int64   `json:"compressed_size_bytes"`
	Ratio           float64 `json:"compression_ratio"`
	ChunkCount      int     `json:"chunk_count"`
	UniqueChunks    int     `json:"unique_chunks"`
	DedupRate       float64 `json:"dedup_rate_percent"`
	CodebookSize    int     `json:"codebook_size"`
	MerkleOverhead  int     `json:"merkle_overhead_bytes"`
	DNAOverhead     float64 `json:"dna_overhead_percent"`
}

// DeltaMetrics contém métricas do tensor delta
type DeltaMetrics struct {
	Timestamp     string  `json:"timestamp"`
	DeltaType     string  `json:"delta_type"`
	DeltaSize     int     `json:"delta_size_bytes"`
	BrainSize     int64   `json:"brain_size_bytes"`
	DeltaRatio    float64 `json:"delta_brain_ratio_percent"`
	NonZeroBytes  int     `json:"non_zero_bytes"`
	Sparsity      float64 `json:"sparsity_percent"`
	ApplyLatency  int64   `json:"apply_latency_ns"`
}

// EntropyMetrics contém medições de entropia de Shannon
type EntropyMetrics struct {
	Timestamp      string    `json:"timestamp"`
	Source         string    `json:"source"`
	ChunkEntropies []float64 `json:"chunk_entropies"`
	MeanEntropy    float64   `json:"mean_entropy"`
	StdEntropy     float64   `json:"std_entropy"`
	MinEntropy     float64   `json:"min_entropy"`
	MaxEntropy     float64   `json:"max_entropy"`
}

// RoutingMetrics contém métricas de multi-brain routing
type RoutingMetrics struct {
	Timestamp      string    `json:"timestamp"`
	NumBrains      int       `json:"num_brains"`
	DecisionTimeNs int64     `json:"decision_time_ns"`
	SelectedBrains []int     `json:"selected_brains"`
	Weights        []float64 `json:"weights"`
	MemoryUsedMB   float64   `json:"memory_used_mb"`
}

// BenchmarkResult é o resultado completo de um benchmark
type BenchmarkResult struct {
	TestName    string      `json:"test_name"`
	Timestamp   string      `json:"timestamp"`
	Duration    int64       `json:"duration_ns"`
	Iterations  int         `json:"iterations"`
	NsPerOp     float64     `json:"ns_per_op"`
	MBPerSec    float64     `json:"mb_per_sec"`
	ExtraData   interface{} `json:"extra_data,omitempty"`
}

// ============================================================================
// FUNÇÕES CORE — SIMULAÇÃO
// ============================================================================

// GenerateSyntheticBrain cria um brain.crom sintético para testes.
// numChunks: número de chunks CDC, chunkSize: tamanho médio de cada chunk em bytes.
func GenerateSyntheticBrain(numChunks int, chunkSize int) *BrainCrom {
	brain := &BrainCrom{
		Header: CromHeader{
			Magic:      "CROM",
			Version:    1,
			Frozen:     false,
			Domain:     "brain",
			ChunkCount: numChunks,
		},
		Chunks:   make([]Chunk, numChunks),
		Codebook: make([]CodeEntry, 0),
	}

	totalOriginal := int64(numChunks * chunkSize)
	totalCompressed := int64(0)
	leaves := make([]string, numChunks)
	codebookMap := make(map[uint64]int)

	for i := 0; i < numChunks; i++ {
		data := make([]byte, chunkSize)
		// Simula dados com padrões (não puramente aleatório)
		// ~70% dos bytes seguem padrão, ~30% são "ruído"
		for j := 0; j < chunkSize; j++ {
			if j%3 != 0 {
				data[j] = byte((i * 7 + j * 13) % 256)
			} else {
				b := make([]byte, 1)
				rand.Read(b)
				data[j] = b[0]
			}
		}

		hash := sha256.Sum256(data)
		hashStr := fmt.Sprintf("%x", hash[:8])
		entropy := ShannonEntropy(data)

		// Simula deduplicação: ~30% dos chunks são referências
		isDelta := false
		deltaRef := -1
		compressedChunkSize := int64(chunkSize)

		hashKey := uint64(hash[0])<<56 | uint64(hash[1])<<48 | uint64(hash[2])<<40
		if _, exists := codebookMap[hashKey]; exists && i > 0 {
			isDelta = true
			deltaRef = codebookMap[hashKey]
			compressedChunkSize = int64(chunkSize / 10) // XOR delta é muito menor
		} else {
			codebookMap[hashKey] = i
			dna := bytesToDNA(data[:min(16, len(data))])
			brain.Codebook = append(brain.Codebook, CodeEntry{
				Hash:     hashKey,
				DNA:      dna,
				RefCount: 1,
			})
		}

		brain.Chunks[i] = Chunk{
			ID:       i,
			Hash:     hashStr,
			Data:     data,
			Size:     chunkSize,
			Entropy:  entropy,
			IsDelta:  isDelta,
			DeltaRef: deltaRef,
		}

		leaves[i] = hashStr
		totalCompressed += compressedChunkSize
	}

	brain.Header.OriginalSize = totalOriginal
	brain.Header.CompressedSize = totalCompressed
	brain.Header.CodebookSize = len(brain.Codebook)

	// Merkle Tree
	root := computeMerkleRoot(leaves)
	brain.Merkle = MerkleTree{
		Root:   root,
		Leaves: leaves,
		Depth:  int(math.Ceil(math.Log2(float64(numChunks)))),
	}
	brain.Header.MerkleRoot = root

	return brain
}

// FreezeBrain congela um brain.crom (marca como read-only)
func FreezeBrain(brain *BrainCrom) {
	brain.Header.Frozen = true
}

// GenerateXORDelta gera um tensor delta XOR aleatório
// sparsity: fração de bytes que são zero (0.0 a 1.0)
func GenerateXORDelta(brain *BrainCrom, sparsity float64) *TensorDelta {
	totalSize := 0
	for _, c := range brain.Chunks {
		totalSize += c.Size
	}

	// Gera delta com esparsificação controlada
	deltaSize := totalSize / 20 // ~5% do brain
	data := make([]byte, deltaSize)
	nonZero := 0

	for i := 0; i < deltaSize; i++ {
		b := make([]byte, 1)
		rand.Read(b)
		// Aplica esparsificação
		threshold := byte(sparsity * 255)
		if b[0] > threshold {
			data[i] = b[0]
			nonZero++
		}
		// else: mantém 0
	}

	nonZeroRatio := float64(nonZero) / float64(deltaSize)

	return &TensorDelta{
		Type:         "xor",
		TargetHash:   brain.Header.MerkleRoot,
		Data:         data,
		Size:         deltaSize,
		NonZeroRatio: nonZeroRatio,
		Sparsity:     1.0 - nonZeroRatio,
	}
}

// GenerateVQDelta gera um tensor delta no espaço Vector Quantization
func GenerateVQDelta(brain *BrainCrom, dimension int) *TensorDelta {
	// Delta VQ é um vetor de offsets sobre centroides do codebook
	numEntries := len(brain.Codebook)
	deltaSize := numEntries * dimension * 4 // float32 por dimensão
	data := make([]byte, deltaSize)

	nonZero := 0
	for i := 0; i < deltaSize; i++ {
		b := make([]byte, 1)
		rand.Read(b)
		// VQ deltas são mais esparsos que XOR
		if b[0] > 200 { // ~78% zero
			data[i] = b[0]
			nonZero++
		}
	}

	return &TensorDelta{
		Type:         "vq",
		TargetHash:   brain.Header.MerkleRoot,
		Data:         data,
		Size:         deltaSize,
		NonZeroRatio: float64(nonZero) / float64(deltaSize),
		Sparsity:     1.0 - float64(nonZero)/float64(deltaSize),
	}
}

// ApplyXORDelta aplica um delta XOR sobre chunks e retorna o resultado
func ApplyXORDelta(chunk []byte, delta []byte) []byte {
	result := make([]byte, len(chunk))
	for i := range chunk {
		result[i] = chunk[i] ^ delta[i%len(delta)]
	}
	return result
}

// ============================================================================
// FUNÇÕES DE MEDIÇÃO
// ============================================================================

// ShannonEntropy calcula a entropia de Shannon de um bloco de bytes (bits/byte)
func ShannonEntropy(data []byte) float64 {
	if len(data) == 0 {
		return 0
	}

	freq := make([]float64, 256)
	for _, b := range data {
		freq[b]++
	}

	n := float64(len(data))
	entropy := 0.0
	for _, f := range freq {
		if f > 0 {
			p := f / n
			entropy -= p * math.Log2(p)
		}
	}
	return entropy
}

// MeasureCompressionMetrics calcula métricas de compressão do brain
func MeasureCompressionMetrics(brain *BrainCrom) CompressionMetrics {
	uniqueChunks := 0
	deltaChunks := 0
	for _, c := range brain.Chunks {
		if !c.IsDelta {
			uniqueChunks++
		} else {
			deltaChunks++
		}
	}

	dedupRate := float64(deltaChunks) / float64(brain.Header.ChunkCount) * 100
	ratio := float64(brain.Header.OriginalSize) / float64(brain.Header.CompressedSize)
	merkleOverhead := brain.Merkle.Depth * len(brain.Merkle.Leaves) * 32 // 32 bytes por hash
	dnaOverhead := float64(len(brain.Codebook)*16) / float64(brain.Header.CompressedSize) * 100

	return CompressionMetrics{
		Timestamp:      time.Now().Format(time.RFC3339),
		OriginalSize:   brain.Header.OriginalSize,
		CompressedSize: brain.Header.CompressedSize,
		Ratio:          ratio,
		ChunkCount:     brain.Header.ChunkCount,
		UniqueChunks:   uniqueChunks,
		DedupRate:      dedupRate,
		CodebookSize:   brain.Header.CodebookSize,
		MerkleOverhead: merkleOverhead,
		DNAOverhead:    dnaOverhead,
	}
}

// MeasureDeltaMetrics calcula métricas do tensor delta
func MeasureDeltaMetrics(delta *TensorDelta, brain *BrainCrom) DeltaMetrics {
	nonZeroBytes := 0
	for _, b := range delta.Data {
		if b != 0 {
			nonZeroBytes++
		}
	}

	// Mede latência de aplicação do delta
	testChunk := brain.Chunks[0].Data
	start := time.Now()
	for i := 0; i < 1000; i++ {
		ApplyXORDelta(testChunk, delta.Data)
	}
	elapsed := time.Since(start).Nanoseconds() / 1000 // por aplicação

	return DeltaMetrics{
		Timestamp:    time.Now().Format(time.RFC3339),
		DeltaType:    delta.Type,
		DeltaSize:    delta.Size,
		BrainSize:    brain.Header.CompressedSize,
		DeltaRatio:   float64(delta.Size) / float64(brain.Header.CompressedSize) * 100,
		NonZeroBytes: nonZeroBytes,
		Sparsity:     (1.0 - float64(nonZeroBytes)/float64(delta.Size)) * 100,
		ApplyLatency: elapsed,
	}
}

// MeasureEntropyMetrics calcula entropia de Shannon de todos os chunks
func MeasureEntropyMetrics(brain *BrainCrom, source string) EntropyMetrics {
	entropies := make([]float64, len(brain.Chunks))
	sum := 0.0
	minE := math.MaxFloat64
	maxE := 0.0

	for i, c := range brain.Chunks {
		e := ShannonEntropy(c.Data)
		entropies[i] = e
		sum += e
		if e < minE {
			minE = e
		}
		if e > maxE {
			maxE = e
		}
	}

	mean := sum / float64(len(entropies))

	// Desvio padrão
	variance := 0.0
	for _, e := range entropies {
		diff := e - mean
		variance += diff * diff
	}
	std := math.Sqrt(variance / float64(len(entropies)))

	return EntropyMetrics{
		Timestamp:      time.Now().Format(time.RFC3339),
		Source:         source,
		ChunkEntropies: entropies,
		MeanEntropy:    mean,
		StdEntropy:     std,
		MinEntropy:     minE,
		MaxEntropy:     maxE,
	}
}

// ============================================================================
// UTILIDADES
// ============================================================================

// bytesToDNA converte bytes para DNA Base-4 (A=00, T=01, C=10, G=11)
func bytesToDNA(data []byte) string {
	bases := []byte{'A', 'T', 'C', 'G'}
	dna := make([]byte, len(data)*4)
	for i, b := range data {
		dna[i*4+0] = bases[(b>>6)&0x03]
		dna[i*4+1] = bases[(b>>4)&0x03]
		dna[i*4+2] = bases[(b>>2)&0x03]
		dna[i*4+3] = bases[b&0x03]
	}
	return string(dna)
}

// computeMerkleRoot calcula a raiz da Merkle Tree a partir das folhas
func computeMerkleRoot(leaves []string) string {
	if len(leaves) == 0 {
		return ""
	}
	if len(leaves) == 1 {
		return leaves[0]
	}

	current := make([]string, len(leaves))
	copy(current, leaves)

	for len(current) > 1 {
		next := make([]string, 0)
		for i := 0; i < len(current); i += 2 {
			var combined string
			if i+1 < len(current) {
				combined = current[i] + current[i+1]
			} else {
				combined = current[i] + current[i]
			}
			hash := sha256.Sum256([]byte(combined))
			next = append(next, fmt.Sprintf("%x", hash[:8]))
		}
		current = next
	}
	return current[0]
}

// SaveJSON salva dados em JSON para o diretório de saída
func SaveJSON(filename string, data interface{}) error {
	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("erro ao criar %s: %w", filename, err)
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(data)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
