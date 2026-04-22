// Package lab17 implements K-Means clustering and Vector Quantization
// in pure Go — zero CGo, zero Python dependencies.
//
// Item 1.2.1: K-Means clustering em Go
// Item 1.2.2: VQ encoder: lookup O(K×D), <10μs para K=256, D=64
//
// This is the Codebook Training engine for the CROM motor,
// enabling Go-native codebook creation without Python/sklearn.
package lab17

import (
	"math"
	"math/rand"
)

// ============================================================================
// K-MEANS CLUSTERING (1.2.1)
// ============================================================================

// KMeans implements K-Means clustering with K-Means++ initialization.
type KMeans struct {
	K         int         // number of clusters
	Dim       int         // dimensionality
	Centroids [][]float64 // K × Dim centroids
	MaxIter   int         // maximum iterations (default: 100)

	// Internal state
	assignments []int       // cluster assignment per data point
	counts      []int       // points per cluster
	sums        [][]float64 // sum of points per cluster (for centroid update)
	converged   bool
	iterations  int
}

// NewKMeans creates a K-Means clusterer.
func NewKMeans(k, dim int) *KMeans {
	km := &KMeans{
		K:       k,
		Dim:     dim,
		MaxIter: 100,
	}
	// Preallocate centroid buffers
	km.Centroids = make([][]float64, k)
	km.sums = make([][]float64, k)
	km.counts = make([]int, k)
	for i := 0; i < k; i++ {
		km.Centroids[i] = make([]float64, dim)
		km.sums[i] = make([]float64, dim)
	}
	return km
}

// Fit trains K-Means on the given data.
//
// Uses K-Means++ initialization for robust centroid placement,
// then iterates Lloyd's algorithm until convergence or maxIter.
func (km *KMeans) Fit(data [][]float64, seed int64) {
	n := len(data)
	if n == 0 {
		return
	}

	rng := rand.New(rand.NewSource(seed))
	km.assignments = make([]int, n)

	// K-Means++ initialization
	km.initPlusPlus(data, rng)

	// Lloyd's algorithm
	km.converged = false
	for iter := 0; iter < km.MaxIter; iter++ {
		km.iterations = iter + 1

		// Assign step: each point → nearest centroid
		changed := 0
		for i, point := range data {
			nearest := km.assign(point)
			if nearest != km.assignments[i] {
				changed++
			}
			km.assignments[i] = nearest
		}

		// Update step: recalculate centroids
		km.updateCentroids(data)

		// Check convergence
		if changed == 0 {
			km.converged = true
			break
		}
	}
}

// initPlusPlus implements K-Means++ initialization.
// Selects initial centroids with probability proportional to D².
func (km *KMeans) initPlusPlus(data [][]float64, rng *rand.Rand) {
	n := len(data)

	// First centroid: random
	idx := rng.Intn(n)
	copy(km.Centroids[0], data[idx])

	// Distance buffer
	dists := make([]float64, n)

	for c := 1; c < km.K; c++ {
		// Calculate D² to nearest existing centroid
		totalDist := 0.0
		for i, point := range data {
			minDist := math.MaxFloat64
			for j := 0; j < c; j++ {
				d := squaredEuclidean(point, km.Centroids[j], km.Dim)
				if d < minDist {
					minDist = d
				}
			}
			dists[i] = minDist
			totalDist += minDist
		}

		// Sample proportional to D²
		target := rng.Float64() * totalDist
		cumSum := 0.0
		chosen := 0
		for i := range dists {
			cumSum += dists[i]
			if cumSum >= target {
				chosen = i
				break
			}
		}
		copy(km.Centroids[chosen%km.K], data[chosen])
		// Safe copy
		for d := 0; d < km.Dim; d++ {
			km.Centroids[c][d] = data[chosen][d]
		}
	}
}

// assign returns the index of the nearest centroid.
func (km *KMeans) assign(point []float64) int {
	bestIdx := 0
	bestDist := math.MaxFloat64

	for c := 0; c < km.K; c++ {
		d := squaredEuclidean(point, km.Centroids[c], km.Dim)
		if d < bestDist {
			bestDist = d
			bestIdx = c
		}
	}
	return bestIdx
}

// updateCentroids recalculates centroids as the mean of assigned points.
func (km *KMeans) updateCentroids(data [][]float64) {
	// Reset sums and counts
	for c := 0; c < km.K; c++ {
		km.counts[c] = 0
		for d := 0; d < km.Dim; d++ {
			km.sums[c][d] = 0
		}
	}

	// Accumulate
	for i, point := range data {
		c := km.assignments[i]
		km.counts[c]++
		for d := 0; d < km.Dim; d++ {
			km.sums[c][d] += point[d]
		}
	}

	// Divide
	for c := 0; c < km.K; c++ {
		if km.counts[c] > 0 {
			inv := 1.0 / float64(km.counts[c])
			for d := 0; d < km.Dim; d++ {
				km.Centroids[c][d] = km.sums[c][d] * inv
			}
		}
	}
}

// Assignments returns the cluster assignment for each data point.
func (km *KMeans) Assignments() []int {
	return km.assignments
}

// Converged returns whether K-Means converged before maxIter.
func (km *KMeans) Converged() bool {
	return km.converged
}

// Iterations returns the number of iterations run.
func (km *KMeans) Iterations() int {
	return km.iterations
}

// Inertia calculates the total within-cluster sum of squared distances.
func (km *KMeans) Inertia(data [][]float64) float64 {
	total := 0.0
	for i, point := range data {
		c := km.assignments[i]
		total += squaredEuclidean(point, km.Centroids[c], km.Dim)
	}
	return total
}

// ============================================================================
// VECTOR QUANTIZER (1.2.2)
// ============================================================================

// VQEncoder wraps a trained K-Means for fast vector quantization.
type VQEncoder struct {
	K         int
	Dim       int
	Centroids [][]float64 // K × Dim codebook
}

// NewVQEncoder creates a VQ encoder from trained centroids.
func NewVQEncoder(centroids [][]float64) *VQEncoder {
	k := len(centroids)
	dim := 0
	if k > 0 {
		dim = len(centroids[0])
	}
	// Deep copy centroids
	copy_ := make([][]float64, k)
	for i := range centroids {
		copy_[i] = make([]float64, dim)
		copy(copy_[i], centroids[i])
	}
	return &VQEncoder{K: k, Dim: dim, Centroids: copy_}
}

// NewVQEncoderFromKMeans creates a VQ encoder from a trained KMeans model.
func NewVQEncoderFromKMeans(km *KMeans) *VQEncoder {
	return NewVQEncoder(km.Centroids)
}

// Encode returns the index of the nearest centroid (codebook lookup).
// Complexity: O(K × D).
func (vq *VQEncoder) Encode(vector []float64) int {
	bestIdx := 0
	bestDist := math.MaxFloat64

	for c := 0; c < vq.K; c++ {
		d := squaredEuclidean(vector, vq.Centroids[c], vq.Dim)
		if d < bestDist {
			bestDist = d
			bestIdx = c
		}
	}
	return bestIdx
}

// Decode returns the centroid for the given index.
// Returns internal buffer — do not modify.
func (vq *VQEncoder) Decode(index int) []float64 {
	if index < 0 || index >= vq.K {
		return nil
	}
	return vq.Centroids[index]
}

// EncodeAll encodes a batch of vectors. Returns indices.
func (vq *VQEncoder) EncodeAll(vectors [][]float64) []int {
	indices := make([]int, len(vectors))
	for i, v := range vectors {
		indices[i] = vq.Encode(v)
	}
	return indices
}

// ReconstructionError computes the mean squared error of VQ encoding.
func (vq *VQEncoder) ReconstructionError(vectors [][]float64) float64 {
	if len(vectors) == 0 {
		return 0
	}
	total := 0.0
	for _, v := range vectors {
		idx := vq.Encode(v)
		total += squaredEuclidean(v, vq.Centroids[idx], vq.Dim)
	}
	return total / float64(len(vectors))
}

// ============================================================================
// UTILITY
// ============================================================================

// squaredEuclidean computes Σ(a[i]-b[i])² for the first dim elements.
func squaredEuclidean(a, b []float64, dim int) float64 {
	sum := 0.0
	for i := 0; i < dim; i++ {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}
