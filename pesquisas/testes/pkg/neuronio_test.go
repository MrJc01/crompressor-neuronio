package neuronio

import (
	"bytes"
	"testing"
)

// ============================================================================
// TESTES DE ENTROPIA
// ============================================================================

func TestShannonEntropy_Zeros(t *testing.T) {
	// Bloco de zeros → entropia = 0 (um único símbolo, nenhuma incerteza)
	data := make([]byte, 1024)
	e := ShannonEntropy(data)
	if e != 0 {
		t.Errorf("Entropia de bloco zero deveria ser 0, got %.6f", e)
	}
}

func TestShannonEntropy_Uniform(t *testing.T) {
	// 256 bytes com cada valor aparecendo exatamente 1 vez → entropia = 8 (máxima)
	data := make([]byte, 256)
	for i := 0; i < 256; i++ {
		data[i] = byte(i)
	}
	e := ShannonEntropy(data)
	if e != 8.0 {
		t.Errorf("Entropia de distribuição uniforme deveria ser 8.0, got %.6f", e)
	}
}

func TestShannonEntropy_Binary(t *testing.T) {
	// Apenas 2 valores (0 e 1), distribuição igual → entropia = 1.0
	data := make([]byte, 1000)
	for i := 0; i < len(data); i++ {
		data[i] = byte(i % 2)
	}
	e := ShannonEntropy(data)
	if e < 0.99 || e > 1.01 {
		t.Errorf("Entropia binária equilibrada deveria ser ~1.0, got %.6f", e)
	}
}

func TestShannonEntropy_Empty(t *testing.T) {
	e := ShannonEntropy([]byte{})
	if e != 0 {
		t.Errorf("Entropia de slice vazio deveria ser 0, got %.6f", e)
	}
}

// ============================================================================
// TESTES DE XOR — REVERSIBILIDADE
// ============================================================================

func TestXOR_Reversibility(t *testing.T) {
	// Propriedade fundamental: A ⊕ B ⊕ B = A
	original := []byte{0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE}
	delta := []byte{0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0}

	modified := ApplyXORDelta(original, delta)
	restored := ApplyXORDelta(modified, delta)

	if !bytes.Equal(original, restored) {
		t.Errorf("XOR não é reversível! original=%x, restored=%x", original, restored)
	}
}

func TestXOR_Identity(t *testing.T) {
	// A ⊕ 0 = A (delta de zeros não altera nada)
	original := []byte{0xDE, 0xAD, 0xBE, 0xEF}
	zeroDelta := make([]byte, len(original))

	result := ApplyXORDelta(original, zeroDelta)
	if !bytes.Equal(original, result) {
		t.Errorf("XOR com zero delta deveria ser identidade! got %x", result)
	}
}

func TestXOR_SelfInverse(t *testing.T) {
	// A ⊕ A = 0 (um dado XOR consigo mesmo é zero)
	data := []byte{0xDE, 0xAD, 0xBE, 0xEF}
	result := ApplyXORDelta(data, data)
	expected := make([]byte, len(data))
	if !bytes.Equal(result, expected) {
		t.Errorf("A ⊕ A deveria ser 0! got %x", result)
	}
}

// ============================================================================
// TESTES DE COMPOSIÇÃO MULTI-DELTA
// ============================================================================

func TestComposeDeltas_Associative(t *testing.T) {
	// (A⊕B)⊕C = A⊕(B⊕C) — associatividade
	a := &TensorDelta{Type: "xor", Data: []byte{0x11, 0x22, 0x33, 0x44}}
	b := &TensorDelta{Type: "xor", Data: []byte{0xAA, 0xBB, 0xCC, 0xDD}}
	c := &TensorDelta{Type: "xor", Data: []byte{0x55, 0x66, 0x77, 0x88}}

	ab := ComposeDeltas(a, b)
	abc1 := ComposeDeltas(ab, c)

	bc := ComposeDeltas(b, c)
	abc2 := ComposeDeltas(a, bc)

	if !bytes.Equal(abc1.Data, abc2.Data) {
		t.Errorf("XOR não é associativo! (A⊕B)⊕C=%x, A⊕(B⊕C)=%x", abc1.Data, abc2.Data)
	}
}

func TestComposeDeltas_Commutative(t *testing.T) {
	// A⊕B = B⊕A — comutatividade
	a := &TensorDelta{Type: "xor", Data: []byte{0x11, 0x22, 0x33}}
	b := &TensorDelta{Type: "xor", Data: []byte{0xAA, 0xBB, 0xCC}}

	ab := ComposeDeltas(a, b)
	ba := ComposeDeltas(b, a)

	if !bytes.Equal(ab.Data, ba.Data) {
		t.Errorf("XOR não é comutativo! A⊕B=%x, B⊕A=%x", ab.Data, ba.Data)
	}
}

func TestComposeDeltas_DoubleApply(t *testing.T) {
	// Aplicar composição é equivalente a aplicar individualmente:
	// chunk ⊕ (delta1 ⊕ delta2) = (chunk ⊕ delta1) ⊕ delta2
	chunk := []byte{0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE}
	d1 := &TensorDelta{Type: "xor", Data: []byte{0x11, 0x22, 0x33, 0x44, 0x55, 0x66}}
	d2 := &TensorDelta{Type: "xor", Data: []byte{0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF}}

	// Caminho 1: compor deltas, depois aplicar
	composed := ComposeDeltas(d1, d2)
	result1 := ApplyXORDelta(chunk, composed.Data)

	// Caminho 2: aplicar d1, depois d2
	step1 := ApplyXORDelta(chunk, d1.Data)
	result2 := ApplyXORDelta(step1, d2.Data)

	if !bytes.Equal(result1, result2) {
		t.Errorf("Composição não é equivalente! path1=%x, path2=%x", result1, result2)
	}
}

// ============================================================================
// TESTES DE DNA
// ============================================================================

func TestDNA_Roundtrip(t *testing.T) {
	// Byte → DNA → Byte deve ser identidade
	original := []byte{0x00, 0xFF, 0xAB, 0xCD, 0x12, 0x34}
	dna := BytesToDNA(original)
	restored := DNAToBytes(dna)

	if !bytes.Equal(original, restored) {
		t.Errorf("DNA roundtrip falhou! original=%x, restored=%x, dna=%s", original, restored, dna)
	}
}

func TestDNA_KnownValues(t *testing.T) {
	// Each byte → 4 DNA bases, 2 bits each (MSB first)
	// 0x00 = 00 00 00 00 → A A A A
	// 0xFF = 11 11 11 11 → G G G G
	// 0x55 = 01 01 01 01 → T T T T
	// 0xAA = 10 10 10 10 → C C C C
	tests := []struct {
		input    byte
		expected string
	}{
		{0x00, "AAAA"},
		{0xFF, "GGGG"},
		{0x55, "TTTT"}, // 01 01 01 01
		{0xAA, "CCCC"}, // 10 10 10 10
	}

	for _, tt := range tests {
		result := BytesToDNA([]byte{tt.input})
		if result != tt.expected {
			t.Errorf("DNA(0x%02X) = %s, expected %s", tt.input, result, tt.expected)
		}
	}
}

// ============================================================================
// TESTES DE MERKLE
// ============================================================================

func TestMerkle_Deterministic(t *testing.T) {
	// Mesmo input → mesmo root
	leaves := []string{"aabb", "ccdd", "eeff", "0011"}
	root1 := ComputeMerkleRoot(leaves)
	root2 := ComputeMerkleRoot(leaves)
	if root1 != root2 {
		t.Errorf("Merkle não é determinístico! %s != %s", root1, root2)
	}
}

func TestMerkle_ChangeDetection(t *testing.T) {
	// Alterar uma folha → root diferente
	leaves1 := []string{"aabb", "ccdd", "eeff", "0011"}
	leaves2 := []string{"aabb", "ccdd", "eeff", "0012"} // 1 bit diferente

	root1 := ComputeMerkleRoot(leaves1)
	root2 := ComputeMerkleRoot(leaves2)

	if root1 == root2 {
		t.Error("Merkle deveria detectar alteração em 1 folha!")
	}
}

func TestMerkle_VerifyBrain(t *testing.T) {
	brain := GenerateSyntheticBrain(100, 128)
	if !VerifyMerkleRoot(brain) {
		t.Error("Merkle deveria verificar brain recém-criado!")
	}

	// Corromper 1 chunk
	brain.Chunks[0].Data[0] ^= 0xFF
	if VerifyMerkleRoot(brain) {
		t.Error("Merkle deveria detectar corrupção!")
	}
}

func TestVerifyChunk(t *testing.T) {
	brain := GenerateSyntheticBrain(10, 64)
	for i, c := range brain.Chunks {
		if !VerifyChunk(&c) {
			t.Errorf("Chunk %d deveria ser válido!", i)
		}
	}

	// Corromper chunk 5
	brain.Chunks[5].Data[0] ^= 0xFF
	if VerifyChunk(&brain.Chunks[5]) {
		t.Error("Chunk corrompido deveria falhar verificação!")
	}
}

// ============================================================================
// TESTES DE DEDUPLICAÇÃO (COMPRESSÃO)
// ============================================================================

func TestCompression_HasDedup(t *testing.T) {
	// 500+ chunks needed to trigger template collisions
	// (10 embed templates × ~50 chunks each = guaranteed repeats)
	brain := GenerateSyntheticBrain(500, 256)
	cm := MeasureCompressionMetrics(brain, "test")

	if cm.DedupRate == 0 {
		t.Errorf("Deveria haver deduplicação > 0%% com dados padronizados, got unique=%d/%d", cm.UniqueChunks, cm.ChunkCount)
	}
	if cm.Ratio <= 1.0 {
		t.Errorf("Compression ratio deveria ser > 1.0, got %.4f", cm.Ratio)
	}
	t.Logf("Compression: ratio=%.2fx, dedup=%.1f%%, unique=%d/%d",
		cm.Ratio, cm.DedupRate, cm.UniqueChunks, cm.ChunkCount)
}

func TestCompression_FrozenFlag(t *testing.T) {
	brain := GenerateSyntheticBrain(10, 64)
	if brain.Header.Frozen {
		t.Error("Brain recém-criado não deveria estar frozen!")
	}
	FreezeBrain(brain)
	if !brain.Header.Frozen {
		t.Error("Brain deveria estar frozen após FreezeBrain!")
	}
}

// ============================================================================
// TESTES DE DELTA SPARSITY
// ============================================================================

func TestDelta_SparsityControl(t *testing.T) {
	brain := GenerateSyntheticBrain(100, 256)

	targets := []struct {
		sparsity    float64
		minExpected float64
		maxExpected float64
	}{
		{0.70, 60, 80},
		{0.85, 75, 95},
		{0.95, 88, 100},
	}

	for _, tt := range targets {
		delta := GenerateXORDelta(brain, tt.sparsity)
		actualSparsity := delta.Sparsity * 100

		if actualSparsity < tt.minExpected || actualSparsity > tt.maxExpected {
			t.Errorf("Sparsity target=%.0f%%: got %.1f%%, expected [%.0f%%, %.0f%%]",
				tt.sparsity*100, actualSparsity, tt.minExpected, tt.maxExpected)
		}
	}
}

// ============================================================================
// BENCHMARKS
// ============================================================================

func BenchmarkXORDelta_512(b *testing.B) {
	chunk := make([]byte, 512)
	delta := make([]byte, 512)
	for i := range chunk {
		chunk[i] = byte(i)
		delta[i] = byte(i * 7)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ApplyXORDelta(chunk, delta)
	}
}

func BenchmarkXORDelta_4096(b *testing.B) {
	chunk := make([]byte, 4096)
	delta := make([]byte, 4096)
	for i := range chunk {
		chunk[i] = byte(i)
		delta[i] = byte(i * 7)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ApplyXORDelta(chunk, delta)
	}
}

func BenchmarkShannonEntropy_512(b *testing.B) {
	data := make([]byte, 512)
	for i := range data {
		data[i] = byte(i * 13)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ShannonEntropy(data)
	}
}

func BenchmarkMerkleRoot_1000(b *testing.B) {
	leaves := make([]string, 1000)
	for i := range leaves {
		leaves[i] = "abcdef1234567890abcdef1234567890"
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ComputeMerkleRoot(leaves)
	}
}

func BenchmarkGenerateBrain_1000(b *testing.B) {
	for i := 0; i < b.N; i++ {
		GenerateSyntheticBrain(1000, 256)
	}
}
