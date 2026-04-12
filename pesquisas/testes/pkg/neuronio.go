// Package neuronio fornece primitivas de teste para as 3 vertentes do crompressor-neuronio.
// Simula operações de XOR Delta, Vector Quantization, entropia de Shannon e
// composição multi-brain, gerando dados mensuráveis em JSON para análise.
package neuronio

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
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
	Header   CromHeader  `json:"header"`
	Chunks   []Chunk     `json:"chunks"`
	Codebook []CodeEntry `json:"codebook"`
	Merkle   MerkleTree  `json:"merkle"`
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
	ID       int     `json:"id"`
	Hash     string  `json:"hash"`
	Data     []byte  `json:"-"`
	Size     int     `json:"size"`
	Entropy  float64 `json:"entropy"`
	IsDelta  bool    `json:"is_delta"`  // true se é referência dedup
	DeltaRef int     `json:"delta_ref"` // ID do chunk referenciado
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
	ModelName       string  `json:"model_name"`
	OriginalSize    int64   `json:"original_size_bytes"`
	CompressedSize  int64   `json:"compressed_size_bytes"`
	Ratio           float64 `json:"compression_ratio"`
	ChunkCount      int     `json:"chunk_count"`
	UniqueChunks    int     `json:"unique_chunks"`
	DedupChunks     int     `json:"dedup_chunks"`
	DedupRate       float64 `json:"dedup_rate_percent"`
	CodebookSize    int     `json:"codebook_size"`
	MerkleOverhead  int     `json:"merkle_overhead_bytes"`
	DNAOverhead     float64 `json:"dna_overhead_percent"`
}

// DeltaMetrics contém métricas do tensor delta
type DeltaMetrics struct {
	Timestamp    string  `json:"timestamp"`
	DeltaType    string  `json:"delta_type"`
	DeltaSize    int     `json:"delta_size_bytes"`
	BrainSize    int64   `json:"brain_size_bytes"`
	DeltaRatio   float64 `json:"delta_brain_ratio_percent"`
	NonZeroBytes int     `json:"non_zero_bytes"`
	Sparsity     float64 `json:"sparsity_percent"`
	ApplyLatency int64   `json:"apply_latency_ns"`
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
	TestName   string      `json:"test_name"`
	Timestamp  string      `json:"timestamp"`
	Duration   int64       `json:"duration_ns"`
	Iterations int         `json:"iterations"`
	NsPerOp    float64     `json:"ns_per_op"`
	MBPerSec   float64     `json:"mb_per_sec"`
	ExtraData  interface{} `json:"extra_data,omitempty"`
}

// ============================================================================
// FUNÇÕES CORE — SIMULAÇÃO REALISTA
// ============================================================================

// GenerateSyntheticBrain cria um brain.crom sintético com padrões de deduplicação
// realistas. Simula como modelos GGUF reais produzem chunks repetitivos:
//   - ~40% dos chunks são blocos de "embedding" (padrão repetitivo → dedup alto)
//   - ~30% dos chunks são blocos de "atenção" (entropia média)
//   - ~30% dos chunks são blocos de "FFN" (alta entropia, pouca dedup)
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
	// Mapa de hash→chunkID para deduplicação
	dedupMap := make(map[string]int)
	uniqueCount := 0
	dedupCount := 0

	// Gera um set finito de "templates" para simular padrões repetitivos
	numEmbedTemplates := 10  // apenas 10 padrões únicos de embedding
	numAttnTemplates := 30   // 30 padrões de atenção
	embedTemplates := make([][]byte, numEmbedTemplates)
	attnTemplates := make([][]byte, numAttnTemplates)

	for i := 0; i < numEmbedTemplates; i++ {
		t := make([]byte, chunkSize)
		// Embedding: padrão muito repetitivo (baixa entropia ~3-4 bits/byte)
		for j := 0; j < chunkSize; j++ {
			t[j] = byte((i*17 + j*3) % 64) // usa apenas 64 valores → ~6 bits/byte
		}
		embedTemplates[i] = t
	}
	for i := 0; i < numAttnTemplates; i++ {
		t := make([]byte, chunkSize)
		// Atenção: padrão com variação moderada (entropia ~5-6 bits/byte)
		for j := 0; j < chunkSize; j++ {
			t[j] = byte((i*31 + j*7) % 160) // usa 160 valores → ~7 bits/byte
		}
		attnTemplates[i] = t
	}

	for i := 0; i < numChunks; i++ {
		data := make([]byte, chunkSize)

		// Decidir tipo de chunk com proporções realistas
		chunkType := i % 10
		switch {
		case chunkType < 4: // 40% embedding (alta dedup — cópias exatas dos templates)
			templateIdx := i % numEmbedTemplates
			copy(data, embedTemplates[templateIdx])
			// SEM variação → gera dedup real (hash idêntico ao template)
		case chunkType < 7: // 30% atenção (dedup moderada — variação por posição)
			templateIdx := i % numAttnTemplates
			copy(data, attnTemplates[templateIdx])
			data[0] = byte(i / numAttnTemplates) // variação leve → poucos duplicados
		default: // 30% FFN (alta entropia, pouca dedup)
			// Dados pseudo-aleatórios baseados em seed determinístico
			seed := uint64(i * 6364136223846793005)
			for j := 0; j < chunkSize; j++ {
				seed = seed*6364136223846793005 + 1442695040888963407
				data[j] = byte(seed >> 33)
			}
		}

		hash := sha256.Sum256(data)
		hashStr := fmt.Sprintf("%x", hash[:16])
		entropy := ShannonEntropy(data)

		// Deduplicação real: se hash já existe, é uma referência
		isDelta := false
		deltaRef := -1
		compressedChunkSize := int64(chunkSize)

		if existing, exists := dedupMap[hashStr]; exists {
			isDelta = true
			deltaRef = existing
			compressedChunkSize = 32 // apenas o hash de referência
			dedupCount++
		} else {
			dedupMap[hashStr] = i
			uniqueCount++
			dna := BytesToDNA(data[:minInt(32, len(data))])
			brain.Codebook = append(brain.Codebook, CodeEntry{
				Hash:     binary.BigEndian.Uint64(hash[:8]),
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
	root := ComputeMerkleRoot(leaves)
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

// GenerateXORDelta gera um tensor delta XOR com esparsificação controlada.
// sparsity: fração de bytes que são zero (0.0 a 1.0)
// targetRatio: tamanho do delta como fração do brain (ex: 0.05 = 5%)
func GenerateXORDelta(brain *BrainCrom, sparsity float64) *TensorDelta {
	// Delta é ~5% do brain comprimido
	deltaSize := int(brain.Header.CompressedSize / 20)
	if deltaSize < 64 {
		deltaSize = 64
	}
	data := make([]byte, deltaSize)
	nonZero := 0

	// Gera delta usando buffer de random mais eficiente
	randBuf := make([]byte, deltaSize)
	rand.Read(randBuf)

	threshold := byte(sparsity * 255)
	for i := 0; i < deltaSize; i++ {
		if randBuf[i] > threshold {
			data[i] = randBuf[i]
			nonZero++
		}
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

// GenerateVQDelta gera um tensor delta no espaço Vector Quantization.
// Em vez de operar bit-a-bit como XOR, opera no espaço do codebook:
// cada offset modifica um centroide do codebook.
func GenerateVQDelta(brain *BrainCrom, dimension int) *TensorDelta {
	// VQ delta: apenas OFFSETS no codebook, não dados raw
	// Cada entrada = (codebook_index uint16, offset_vector []float16)
	// Muito mais compacto: ~20% das entradas são modificadas
	numEntries := len(brain.Codebook)
	modifiedEntries := numEntries / 5 // modifica 20% do codebook
	if modifiedEntries < 1 {
		modifiedEntries = 1
	}

	// Cada modificação: 2 bytes (index) + dimension/4 bytes (quantized offsets)
	entrySize := 2 + dimension/4
	deltaSize := modifiedEntries * entrySize
	data := make([]byte, deltaSize)

	nonZero := 0
	randBuf := make([]byte, deltaSize)
	rand.Read(randBuf)

	for i := 0; i < deltaSize; i++ {
		if randBuf[i] > 180 { // ~29% não-zero
			data[i] = randBuf[i]
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

// ApplyXORDelta aplica um delta XOR sobre um chunk e retorna o resultado.
// Propriedade fundamental: A ⊕ B ⊕ B = A (reversível)
func ApplyXORDelta(chunk []byte, delta []byte) []byte {
	result := make([]byte, len(chunk))
	for i := range chunk {
		result[i] = chunk[i] ^ delta[i%len(delta)]
	}
	return result
}

// ComposeDeltas combina dois deltas XOR via associatividade: (A⊕B)⊕C = A⊕(B⊕C)
// Isso permite acumular deltas sem precisar do brain original.
func ComposeDeltas(delta1, delta2 *TensorDelta) *TensorDelta {
	// Usa o maior como base
	maxLen := len(delta1.Data)
	if len(delta2.Data) > maxLen {
		maxLen = len(delta2.Data)
	}

	composed := make([]byte, maxLen)
	nonZero := 0
	for i := 0; i < maxLen; i++ {
		var b1, b2 byte
		if i < len(delta1.Data) {
			b1 = delta1.Data[i]
		}
		if i < len(delta2.Data) {
			b2 = delta2.Data[i]
		}
		composed[i] = b1 ^ b2
		if composed[i] != 0 {
			nonZero++
		}
	}

	return &TensorDelta{
		Type:         "composed",
		TargetHash:   delta1.TargetHash,
		Data:         composed,
		Size:         maxLen,
		NonZeroRatio: float64(nonZero) / float64(maxLen),
		Sparsity:     1.0 - float64(nonZero)/float64(maxLen),
	}
}

// ============================================================================
// FUNÇÕES DE MEDIÇÃO
// ============================================================================

// ShannonEntropy calcula a entropia de Shannon de um bloco de bytes (bits/byte).
// H = -Σ p(x) log₂ p(x), onde p(x) é a frequência do byte x.
// Máximo teórico: 8 bits/byte (distribuição uniforme sobre 256 valores).
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
func MeasureCompressionMetrics(brain *BrainCrom, modelName string) CompressionMetrics {
	uniqueChunks := 0
	dedupChunks := 0
	for _, c := range brain.Chunks {
		if !c.IsDelta {
			uniqueChunks++
		} else {
			dedupChunks++
		}
	}

	dedupRate := float64(dedupChunks) / float64(brain.Header.ChunkCount) * 100
	ratio := float64(brain.Header.OriginalSize) / float64(brain.Header.CompressedSize)
	merkleOverhead := brain.Merkle.Depth * len(brain.Merkle.Leaves) * 32 // 32 bytes por hash
	dnaOverhead := float64(len(brain.Codebook)*32) / float64(brain.Header.CompressedSize) * 100

	return CompressionMetrics{
		Timestamp:      time.Now().Format(time.RFC3339),
		ModelName:      modelName,
		OriginalSize:   brain.Header.OriginalSize,
		CompressedSize: brain.Header.CompressedSize,
		Ratio:          ratio,
		ChunkCount:     brain.Header.ChunkCount,
		UniqueChunks:   uniqueChunks,
		DedupChunks:    dedupChunks,
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

	// Mede latência de aplicação do delta (1000 iterações → média)
	testChunk := brain.Chunks[0].Data
	start := time.Now()
	for i := 0; i < 1000; i++ {
		ApplyXORDelta(testChunk, delta.Data)
	}
	elapsed := time.Since(start).Nanoseconds() / 1000

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
// MERKLE — VERIFICAÇÃO DE INTEGRIDADE
// ============================================================================

// VerifyMerkleRoot recalcula o Merkle root e verifica contra o armazenado
func VerifyMerkleRoot(brain *BrainCrom) bool {
	leaves := make([]string, len(brain.Chunks))
	for i, c := range brain.Chunks {
		hash := sha256.Sum256(c.Data)
		leaves[i] = fmt.Sprintf("%x", hash[:16])
	}
	computed := ComputeMerkleRoot(leaves)
	return computed == brain.Header.MerkleRoot
}

// VerifyChunk verifica integridade de um chunk individual
func VerifyChunk(chunk *Chunk) bool {
	hash := sha256.Sum256(chunk.Data)
	return fmt.Sprintf("%x", hash[:16]) == chunk.Hash
}

// ComputeMerkleRoot calcula a raiz da Merkle Tree a partir das folhas
func ComputeMerkleRoot(leaves []string) string {
	if len(leaves) == 0 {
		return ""
	}
	if len(leaves) == 1 {
		return leaves[0]
	}

	current := make([]string, len(leaves))
	copy(current, leaves)

	for len(current) > 1 {
		next := make([]string, 0, (len(current)+1)/2)
		for i := 0; i < len(current); i += 2 {
			var combined string
			if i+1 < len(current) {
				combined = current[i] + current[i+1]
			} else {
				combined = current[i] + current[i]
			}
			hash := sha256.Sum256([]byte(combined))
			next = append(next, fmt.Sprintf("%x", hash[:16]))
		}
		current = next
	}
	return current[0]
}

// ============================================================================
// SERIALIZAÇÃO .crom
// ============================================================================

// WriteCrom serializa um brain para arquivo JSON (simulação do formato .crom)
func WriteCrom(filename string, brain *BrainCrom) error {
	// Salva header + codebook + merkle (sem dados dos chunks)
	type CromFile struct {
		Header   CromHeader  `json:"header"`
		Codebook []CodeEntry `json:"codebook"`
		Merkle   MerkleTree  `json:"merkle"`
		ChunkMeta []struct {
			ID      int     `json:"id"`
			Hash    string  `json:"hash"`
			Size    int     `json:"size"`
			Entropy float64 `json:"entropy"`
			IsDelta bool    `json:"is_delta"`
			DeltaRef int    `json:"delta_ref"`
		} `json:"chunk_meta"`
	}

	cf := CromFile{
		Header:   brain.Header,
		Codebook: brain.Codebook,
		Merkle:   brain.Merkle,
	}
	for _, c := range brain.Chunks {
		cf.ChunkMeta = append(cf.ChunkMeta, struct {
			ID      int     `json:"id"`
			Hash    string  `json:"hash"`
			Size    int     `json:"size"`
			Entropy float64 `json:"entropy"`
			IsDelta bool    `json:"is_delta"`
			DeltaRef int    `json:"delta_ref"`
		}{c.ID, c.Hash, c.Size, c.Entropy, c.IsDelta, c.DeltaRef})
	}

	return SaveJSON(filename, cf)
}

// ============================================================================
// UTILIDADES
// ============================================================================

// BytesToDNA converte bytes para DNA Base-4 (A=00, T=01, C=10, G=11)
func BytesToDNA(data []byte) string {
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

// DNAToBytes converte DNA Base-4 de volta para bytes
func DNAToBytes(dna string) []byte {
	if len(dna)%4 != 0 {
		return nil
	}
	result := make([]byte, len(dna)/4)
	baseMap := map[byte]byte{'A': 0, 'T': 1, 'C': 2, 'G': 3}
	for i := 0; i < len(dna); i += 4 {
		var b byte
		b |= baseMap[dna[i+0]] << 6
		b |= baseMap[dna[i+1]] << 4
		b |= baseMap[dna[i+2]] << 2
		b |= baseMap[dna[i+3]]
		result[i/4] = b
	}
	return result
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

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
