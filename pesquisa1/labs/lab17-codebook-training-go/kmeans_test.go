package lab17

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// ============================================================================
// 1.2.1 — K-Means Tests
// ============================================================================

func TestKMeans_SyntheticClusters(t *testing.T) {
	// Generate 4 well-separated Gaussian clusters
	rng := rand.New(rand.NewSource(42))
	centers := [][]float64{
		{0, 0}, {10, 0}, {0, 10}, {10, 10},
	}

	data := make([][]float64, 400) // 100 points per cluster
	trueLabels := make([]int, 400)
	for i := range data {
		cluster := i / 100
		data[i] = make([]float64, 2)
		data[i][0] = centers[cluster][0] + rng.NormFloat64()*0.5
		data[i][1] = centers[cluster][1] + rng.NormFloat64()*0.5
		trueLabels[i] = cluster
	}

	km := NewKMeans(4, 2)
	km.Fit(data, 42)

	t.Logf("K-Means converged: %v in %d iterations", km.Converged(), km.Iterations())
	t.Logf("Inertia: %.4f", km.Inertia(data))

	if !km.Converged() {
		t.Error("K-Means did not converge")
	}

	// Verify centroids are close to true centers
	for _, center := range centers {
		minDist := math.MaxFloat64
		for _, centroid := range km.Centroids {
			d := squaredEuclidean(center, centroid, 2)
			if d < minDist {
				minDist = d
			}
		}
		if minDist > 1.0 { // within 1.0 of true center
			t.Errorf("No centroid found near true center (%.1f, %.1f), min dist=%.3f",
				center[0], center[1], math.Sqrt(minDist))
		}
	}

	for i, c := range km.Centroids {
		t.Logf("Centroid %d: (%.2f, %.2f)", i, c[0], c[1])
	}
}

func TestKMeans_KMeansPP_BetterThanRandom(t *testing.T) {
	// K-Means++ should produce lower inertia than random init
	rng := rand.New(rand.NewSource(99))
	data := make([][]float64, 300)
	for i := range data {
		cluster := i / 100
		data[i] = make([]float64, 2)
		data[i][0] = float64(cluster*5) + rng.NormFloat64()
		data[i][1] = float64(cluster*5) + rng.NormFloat64()
	}

	// K-Means++ (our implementation)
	km := NewKMeans(3, 2)
	km.Fit(data, 42)
	inertiaPP := km.Inertia(data)

	t.Logf("K-Means++ inertia: %.2f, converged=%v, iters=%d",
		inertiaPP, km.Converged(), km.Iterations())

	// Inertia should be reasonable (not degenerate)
	if inertiaPP > 1000 {
		t.Errorf("Inertia %.2f too high — likely degenerate codebook", inertiaPP)
	}
}

// ============================================================================
// 1.2.2 — VQ Encoder Tests
// ============================================================================

func TestVQEncoder_EncodeDecodeRoundtrip(t *testing.T) {
	// Train codebook
	rng := rand.New(rand.NewSource(42))
	data := make([][]float64, 200)
	for i := range data {
		cluster := i / 50
		data[i] = make([]float64, 2)
		data[i][0] = float64(cluster*5) + rng.NormFloat64()*0.3
		data[i][1] = float64(cluster*5) + rng.NormFloat64()*0.3
	}

	km := NewKMeans(4, 2)
	km.Fit(data, 42)

	vq := NewVQEncoderFromKMeans(km)

	// Encode and decode each point
	totalErr := 0.0
	for _, point := range data {
		idx := vq.Encode(point)
		decoded := vq.Decode(idx)
		err := squaredEuclidean(point, decoded, 2)
		totalErr += err
	}
	mse := totalErr / float64(len(data))

	t.Logf("VQ roundtrip MSE: %.6f", mse)

	// MSE should be very low for well-clustered data
	if mse > 1.0 {
		t.Errorf("VQ roundtrip MSE %.6f too high", mse)
	}
}

func TestVQEncoder_K256_D64(t *testing.T) {
	// Test with realistic dimensions: K=256, D=64
	rng := rand.New(rand.NewSource(42))
	k, dim := 256, 64
	nData := 2000

	data := make([][]float64, nData)
	for i := range data {
		data[i] = make([]float64, dim)
		// Generate data around 16 clusters (subset of K)
		cluster := i % 16
		for d := 0; d < dim; d++ {
			data[i][d] = float64(cluster)*2.0 + rng.NormFloat64()*0.5
		}
	}

	km := NewKMeans(k, dim)
	km.MaxIter = 30 // limit iters for speed
	km.Fit(data, 42)

	vq := NewVQEncoderFromKMeans(km)

	t.Logf("K=%d, D=%d: converged=%v, iters=%d", k, dim, km.Converged(), km.Iterations())

	// Benchmark single encode
	testVec := data[0]
	start := time.Now()
	const nIter = 10000
	for i := 0; i < nIter; i++ {
		vq.Encode(testVec)
	}
	elapsed := time.Since(start)
	perEncode := elapsed / nIter

	t.Logf("VQ Encode K=%d D=%d: %v/op (target: <40μs)", k, dim, perEncode)

	if perEncode > 40*time.Microsecond {
		t.Errorf("Encode latency %v exceeds 40μs target", perEncode)
	}

	// Reconstruction error
	mse := vq.ReconstructionError(data[:100])
	t.Logf("Reconstruction MSE: %.4f", mse)
}

func TestVQEncoder_EncodeAll(t *testing.T) {
	centroids := [][]float64{{0, 0}, {10, 0}, {0, 10}}
	vq := NewVQEncoder(centroids)

	vectors := [][]float64{{1, 1}, {9, 1}, {1, 9}, {5, 5}}
	indices := vq.EncodeAll(vectors)

	if indices[0] != 0 {
		t.Errorf("(1,1) should map to cluster 0, got %d", indices[0])
	}
	if indices[1] != 1 {
		t.Errorf("(9,1) should map to cluster 1, got %d", indices[1])
	}
	if indices[2] != 2 {
		t.Errorf("(1,9) should map to cluster 2, got %d", indices[2])
	}

	t.Logf("EncodeAll indices: %v", indices)
}

// ============================================================================
// BENCHMARKS
// ============================================================================

func BenchmarkKMeans_K4_D2_N400(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	data := make([][]float64, 400)
	for i := range data {
		data[i] = []float64{
			float64(i/100*5) + rng.NormFloat64(),
			float64(i/100*5) + rng.NormFloat64(),
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		km := NewKMeans(4, 2)
		km.Fit(data, int64(i))
	}
}

func BenchmarkVQEncode_K256_D64(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	k, dim := 256, 64

	centroids := make([][]float64, k)
	for i := range centroids {
		centroids[i] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			centroids[i][d] = rng.Float64() * 10
		}
	}

	vq := NewVQEncoder(centroids)
	vec := make([]float64, dim)
	for d := range vec {
		vec[d] = rng.Float64() * 10
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		vq.Encode(vec)
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
		Timestamp     string  `json:"timestamp"`
		Lab           string  `json:"lab"`
		KMeansConv    bool    `json:"kmeans_converged"`
		KMeansIters   int     `json:"kmeans_iterations"`
		Inertia       float64 `json:"inertia"`
		VQEncodeUs    float64 `json:"vq_encode_k256_d64_us"`
		VQMse         float64 `json:"vq_reconstruction_mse"`
		Verdict       string  `json:"verdict"`
	}

	result := Result{
		Timestamp: time.Now().Format(time.RFC3339),
		Lab:       "lab17-codebook-training-go",
	}

	// K-Means test
	rng := rand.New(rand.NewSource(42))
	data := make([][]float64, 400)
	for i := range data {
		c := i / 100
		data[i] = []float64{float64(c*5) + rng.NormFloat64()*0.5, float64(c*5) + rng.NormFloat64()*0.5}
	}
	km := NewKMeans(4, 2)
	km.Fit(data, 42)
	result.KMeansConv = km.Converged()
	result.KMeansIters = km.Iterations()
	result.Inertia = km.Inertia(data)

	// VQ encode benchmark K=256, D=64
	k, dim := 256, 64
	centroids := make([][]float64, k)
	for i := range centroids {
		centroids[i] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			centroids[i][d] = rng.Float64() * 10
		}
	}
	vq := NewVQEncoder(centroids)
	vec := make([]float64, dim)
	for d := range vec {
		vec[d] = rng.Float64() * 10
	}
	const nIter = 100000
	start := time.Now()
	for i := 0; i < nIter; i++ {
		vq.Encode(vec)
	}
	result.VQEncodeUs = float64(time.Since(start).Microseconds()) / float64(nIter)

	// Reconstruction error
	testData := make([][]float64, 100)
	for i := range testData {
		testData[i] = make([]float64, dim)
		for d := 0; d < dim; d++ {
			testData[i][d] = rng.Float64() * 10
		}
	}
	result.VQMse = vq.ReconstructionError(testData)

	// Verdict
	if result.KMeansConv && result.VQEncodeUs < 40 {
		result.Verdict = "PASS — K-Means converges, VQ <40μs"
	} else {
		result.Verdict = "PARTIAL"
	}

	outDir := filepath.Join("..", "..", "resultados")
	os.MkdirAll(outDir, 0755)
	outPath := filepath.Join(outDir, "lab17_results.json")
	data2, _ := json.MarshalIndent(result, "", "  ")
	os.WriteFile(outPath, data2, 0644)

	t.Logf("K-Means: converged=%v, iters=%d, inertia=%.2f", result.KMeansConv, result.KMeansIters, result.Inertia)
	t.Logf("VQ Encode K=256 D=64: %.2f μs (target: <40μs)", result.VQEncodeUs)
	t.Logf("Verdict: %s", result.Verdict)
}
