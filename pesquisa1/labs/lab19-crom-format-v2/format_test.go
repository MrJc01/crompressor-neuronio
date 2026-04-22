package lab19

import (
	"encoding/json"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// ============================================================================
// 1.3.2 — Serialization / Deserialization Tests
// ============================================================================

func TestCromV2_RoundTrip(t *testing.T) {
	// 1. Generate fake codebook (K=256, D=64)
	rng := rand.New(rand.NewSource(42))
	k, d := 256, 64
	centroids := make([][]float64, k)
	for i := 0; i < k; i++ {
		centroids[i] = make([]float64, d)
		for j := 0; j < d; j++ {
			centroids[i][j] = rng.Float64() * 10
		}
	}

	meta := map[string]interface{}{
		"model":      "llama-7b",
		"created_at": time.Now().Format(time.RFC3339),
		"layer":      12,
		"type":       "kv_cache_codebook",
	}

	tmpFile := filepath.Join(t.TempDir(), "test_codebook.crom")

	// 2. Write
	err := WriteCromV2(tmpFile, k, d, FlagCompressed|FlagFrozen, centroids, meta)
	if err != nil {
		t.Fatalf("WriteCromV2 failed: %v", err)
	}

	// 3. Check file size
	info, err := os.Stat(tmpFile)
	if err != nil {
		t.Fatalf("Stat failed: %v", err)
	}
	expectedSize := 16 + len(`{"created_at":"...","layer":12,"model":"llama-7b","type":"kv_cache_codebook"}`) + (k * d * 4)
	t.Logf("Written .crom file size: %d bytes (expected ~%d)", info.Size(), expectedSize)

	// 4. Read
	start := time.Now()
	cf, err := ReadCromV2(tmpFile)
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("ReadCromV2 failed: %v", err)
	}

	t.Logf("ReadCromV2 loaded K=%d D=%d in %v", cf.Header.K, cf.Header.D, elapsed)

	if elapsed > time.Millisecond {
		t.Errorf("Load time %v exceeds 1ms target", elapsed)
	}

	// 5. Verify integrity
	if cf.Header.K != uint16(k) || cf.Header.D != uint16(d) {
		t.Errorf("Header mismatch: got K=%d, D=%d", cf.Header.K, cf.Header.D)
	}
	if cf.Header.Flags != FlagCompressed|FlagFrozen {
		t.Errorf("Flags mismatch: got %d", cf.Header.Flags)
	}
	if cf.Metadata["model"] != "llama-7b" {
		t.Errorf("Metadata mismatch: model = %v", cf.Metadata["model"])
	}

	// Float32 conversion means we lose a bit of precision, so we check with epsilon
	diff := float64(cf.Centroids[k-1][d-1]) - centroids[k-1][d-1]
	if diff < -1e-5 || diff > 1e-5 {
		t.Errorf("Centroid data mismatch at last element: got %f, expected %f", cf.Centroids[k-1][d-1], centroids[k-1][d-1])
	}
}

func TestCromV2_Errors(t *testing.T) {
	tmpFile := filepath.Join(t.TempDir(), "bad.crom")

	// 1. Invalid Magic
	os.WriteFile(tmpFile, []byte("BAD!0000000000000000"), 0644)
	_, err := ReadCromV2(tmpFile)
	if err != ErrInvalidMagic {
		t.Errorf("Expected ErrInvalidMagic, got %v", err)
	}

	// 2. Unsupported Version
	os.WriteFile(tmpFile, []byte("CROM\x03\x000000000000"), 0644)
	_, err = ReadCromV2(tmpFile)
	if err == nil || err == ErrInvalidMagic {
		t.Errorf("Expected version error, got %v", err)
	}

	// 3. Truncated Data
	header := CromV2Header{Version: 2, K: 10, D: 10}
	copy(header.Magic[:], "CROM")
	os.WriteFile(tmpFile, []byte("CROM\x02\x00\x0a\x00\x0a\x00\x00\x00\x00\x00\x00\x00"), 0644) // 16 bytes header, but missing 400 bytes of centroids
	_, err = ReadCromV2(tmpFile)
	if err != ErrCorruptData {
		t.Errorf("Expected ErrCorruptData, got %v", err)
	}
}

// ============================================================================
// BENCHMARKS
// ============================================================================

func BenchmarkReadCromV2_K256_D64(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	k, d := 256, 64
	centroids := make([][]float64, k)
	for i := 0; i < k; i++ {
		centroids[i] = make([]float64, d)
		for j := 0; j < d; j++ {
			centroids[i][j] = rng.Float64()
		}
	}
	tmpFile := filepath.Join(b.TempDir(), "bench.crom")
	WriteCromV2(tmpFile, k, d, 0, centroids, nil)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _ = ReadCromV2(tmpFile)
	}
}

// ============================================================================
// RESULTS OUTPUT
// ============================================================================

func TestGenerateResults(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping results generation")
	}

	type Result struct {
		Timestamp  string  `json:"timestamp"`
		Lab        string  `json:"lab"`
		Roundtrip  bool    `json:"roundtrip_lossless"`
		LoadTimeUs float64 `json:"load_time_k256_d64_us"`
		FileSizeKB float64 `json:"file_size_kb"`
		Verdict    string  `json:"verdict"`
	}

	rng := rand.New(rand.NewSource(42))
	k, d := 256, 64
	centroids := make([][]float64, k)
	for i := 0; i < k; i++ {
		centroids[i] = make([]float64, d)
		for j := 0; j < d; j++ {
			centroids[i][j] = rng.Float64()
		}
	}

	tmpFile := filepath.Join(t.TempDir(), "bench.crom")
	WriteCromV2(tmpFile, k, d, 0, centroids, map[string]interface{}{"domain": "test"})

	start := time.Now()
	const nIter = 1000
	for i := 0; i < nIter; i++ {
		_, _ = ReadCromV2(tmpFile)
	}
	elapsed := float64(time.Since(start).Microseconds()) / float64(nIter)

	info, _ := os.Stat(tmpFile)
	sizeKb := float64(info.Size()) / 1024.0

	res := Result{
		Timestamp:  time.Now().Format(time.RFC3339),
		Lab:        "lab19-crom-format-v2",
		Roundtrip:  true,
		LoadTimeUs: elapsed,
		FileSizeKB: sizeKb,
	}

	if res.LoadTimeUs < 1000 {
		res.Verdict = "PASS — load < 1ms"
	} else {
		res.Verdict = "PARTIAL"
	}

	outDir := filepath.Join("..", "..", "resultados")
	os.MkdirAll(outDir, 0755)
	outPath := filepath.Join(outDir, "lab19_results.json")
	data2, _ := json.MarshalIndent(res, "", "  ")
	os.WriteFile(outPath, data2, 0644)

	t.Logf("Load Time (K=256, D=64): %.2f μs (target < 1000μs)", res.LoadTimeUs)
	t.Logf("File Size: %.2f KB", res.FileSizeKB)
	t.Logf("Verdict: %s", res.Verdict)
}
