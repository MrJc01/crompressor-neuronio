// BranchManager implementation using arithmetic deltas with preallocated buffers.
//
// Item 1.1.3: Implementar BranchManager usando DeltaBranchStore existente
// Item 1.1.4: Implementar Decision com weighted collapse por variância
//
// Critério de Sucesso:
//   - 15 branches em <1ms
//   - Agente navega ambiente 1D
package lab13

import (
	"math"
	"math/rand"
)

// CROMBranchManager manages branch exploration using arithmetic deltas.
//
// Key design decisions:
//   - Preallocated slice pool (not map) — zero allocations on hot path
//   - Perturbation scale increases with depth for hierarchical exploration
//   - Collapse uses squared Euclidean distance as free energy proxy
//
// The branch states are stored as full float64 slices (not XOR byte deltas)
// because floating-point arithmetic doesn't compose well with XOR.
// For byte-level storage, the pesquisa0.DeltaBranchStore remains the standard.
type CROMBranchManager struct {
	branches     []BranchResult // preallocated branch pool
	branchStates [][]float64    // preallocated state buffers
	count        int            // active branch count
	dim          int
	maxBranches  int
	perturbScale float64 // base perturbation scale
	rng          *rand.Rand
}

// NewCROMBranchManager creates a branch manager with preallocated buffers.
//
// Parameters:
//   - dim: state dimensionality
//   - maxBranches: maximum branches (depth × width)
//   - perturbScale: base perturbation magnitude
//   - seed: RNG seed for reproducibility
func NewCROMBranchManager(dim, maxBranches int, perturbScale float64, seed int64) *CROMBranchManager {
	bm := &CROMBranchManager{
		branches:     make([]BranchResult, maxBranches),
		branchStates: make([][]float64, maxBranches),
		dim:          dim,
		maxBranches:  maxBranches,
		perturbScale: perturbScale,
		rng:          rand.New(rand.NewSource(seed)),
	}

	// Preallocate state buffers — each branch gets its own slice
	for i := range bm.branchStates {
		bm.branchStates[i] = make([]float64, dim)
		bm.branches[i].State = bm.branchStates[i]
	}

	return bm
}

// Explore generates depth×width branches from the given state.
//
// Each level of depth applies increasingly large perturbations:
//
//	level 1: perturbScale × 1.0 (nearby futures)
//	level 2: perturbScale × 2.0 (medium-range)
//	level 3: perturbScale × 3.0 (distant futures)
//
// This creates a hierarchical exploration tree similar to MCTS.
// Returns slice from internal pool — valid until next Explore call.
func (bm *CROMBranchManager) Explore(state []float64, depth, width int) []BranchResult {
	total := depth * width
	if total > bm.maxBranches {
		total = bm.maxBranches
	}

	bm.count = 0
	for d := 0; d < depth && bm.count < total; d++ {
		scale := bm.perturbScale * float64(d+1) // increasing perturbation with depth
		for w := 0; w < width && bm.count < total; w++ {
			idx := bm.count
			// Copy base state and apply Gaussian perturbation
			for i := 0; i < bm.dim && i < len(state); i++ {
				bm.branchStates[idx][i] = state[i] + bm.rng.NormFloat64()*scale
			}
			bm.branches[idx].ID = idx
			bm.branches[idx].Depth = d + 1
			bm.branches[idx].FreeEnergy = 0 // calculated during collapse
			bm.count++
		}
	}

	return bm.branches[:bm.count]
}

// Collapse evaluates free energy for each branch and returns the best one.
//
// Free energy approximation: F(branch) = Σ(state[i] - target[i])²
// This is the squared Euclidean distance to the target — a proxy for
// prediction error in Active Inference (minimizing variational free energy).
//
// Returns the branch with minimum F and its index.
func (bm *CROMBranchManager) Collapse(target []float64) (BranchResult, int) {
	if bm.count == 0 {
		return BranchResult{}, -1
	}

	bestIdx := 0
	bestFE := math.MaxFloat64

	for i := 0; i < bm.count; i++ {
		fe := 0.0
		for j := 0; j < bm.dim && j < len(target); j++ {
			d := bm.branches[i].State[j] - target[j]
			fe += d * d
		}
		bm.branches[i].FreeEnergy = fe

		if fe < bestFE {
			bestFE = fe
			bestIdx = i
		}
	}

	return bm.branches[bestIdx], bestIdx
}

// BranchCount returns the number of active branches.
func (bm *CROMBranchManager) BranchCount() int {
	return bm.count
}
