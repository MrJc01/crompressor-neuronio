// Package lab13 implements the CROM Agent v2 — a complete Active Inference
// agent in pure Go targeting <0.5ms per step.
//
// Pipeline: Observation → WorldModel → BranchManager → Decision → Firewall → Action
//
// This is the Go-native evolution of the Python prototype from pesquisa0
// (blitz_final.py, 4.77ms/step → target <0.5ms/step, ≥10x speedup).
//
// Items covered:
//   1.1.1 - Interfaces Go (Agent, WorldModel, BranchManager, Firewall)
//   1.1.4 - Decision with weighted collapse
//   1.1.7 - Full pipeline integration
package lab13

import (
	"fmt"
	"math"
)

// ============================================================================
// INTERFACES (1.1.1)
// ============================================================================

// Agent is the main CROM agent interface.
type Agent interface {
	// Step processes one observation and returns an action.
	Step(observation []float64) (action []float64, err error)
	// GetFreeEnergy returns the current free energy of the agent.
	GetFreeEnergy() float64
}

// WorldModel predicts future states and updates with observations.
type WorldModel interface {
	// Predict estimates the next state given current state and optional action.
	Predict(state, action []float64) []float64
	// Update corrects the model with a real observation.
	Update(observation []float64)
	// Error returns the average recent prediction error.
	Error() float64
	// GetState returns the current estimated state (internal buffer, do not modify).
	GetState() []float64
}

// BranchManager explores future states and collapses to the best one.
type BranchManager interface {
	// Explore generates branches from the given state.
	Explore(state []float64, depth, width int) []BranchResult
	// Collapse selects the best branch minimizing distance to target.
	Collapse(target []float64) (BranchResult, int)
	// BranchCount returns the number of active branches.
	BranchCount() int
}

// Firewall checks predictions for anomalies and signs decisions.
type Firewall interface {
	// Check returns whether the prediction is safe.
	Check(prediction, reference []float64) (safe bool, confidence float64)
	// Sign signs data with Ed25519.
	Sign(data []byte) ([]byte, error)
	// Verify verifies a signature.
	Verify(data, signature []byte) bool
	// Stats returns firewall statistics.
	Stats() FirewallStats
}

// BranchResult represents a single explored future state.
type BranchResult struct {
	ID         int
	State      []float64
	Depth      int
	FreeEnergy float64
}

// FirewallStats tracks firewall activity.
type FirewallStats struct {
	TotalChecks   int
	BlockedCount  int
	PassedCount   int
	SignedCount   int
	VerifiedCount int
}

// StepResult contains detailed information about one agent step.
type StepResult struct {
	Action       []float64
	FreeEnergy   float64
	BranchCount  int
	BestBranchID int
	Blocked      bool
	PredError    float64
}

// ============================================================================
// CROM AGENT v2 (1.1.7)
// ============================================================================

// CROMAgent is the concrete implementation integrating all components.
// Uses concrete types internally for zero-overhead (no interface dispatch on hot path).
type CROMAgent struct {
	wm       *EMAWorldModel
	branches *CROMBranchManager
	fw       *ThresholdFirewall

	// Configuration
	dim      int
	target   []float64 // goal state for navigation
	depth    int       // branch exploration depth
	width    int       // branches per depth level

	// State tracking
	freeEnergy float64
	stepCount  int

	// Preallocated buffers (zero-alloc hot path)
	actionBuf []float64
}

// AgentConfig configures the CROM Agent v2.
type AgentConfig struct {
	Dim           int       // state dimensionality
	Target        []float64 // goal state
	Alpha         float64   // EMA smoothing factor (default: 0.3)
	Depth         int       // branch exploration depth (default: 3)
	Width         int       // branches per depth level (default: 5)
	PerturbScale  float64   // branch perturbation scale (default: 1.0)
	FirewallThresh float64  // firewall error threshold (default: 5.0)
	Seed          int64     // RNG seed for reproducibility
}

// DefaultConfig returns sensible defaults for a 1D navigation agent.
func DefaultConfig() AgentConfig {
	return AgentConfig{
		Dim:            1,
		Target:         []float64{10.0},
		Alpha:          0.3,
		Depth:          3,
		Width:          5,
		PerturbScale:   1.0,
		FirewallThresh: 5.0,
		Seed:           42,
	}
}

// NewCROMAgent creates a new CROM Agent v2 with the given configuration.
func NewCROMAgent(cfg AgentConfig) (*CROMAgent, error) {
	if cfg.Dim <= 0 {
		return nil, fmt.Errorf("dim must be > 0, got %d", cfg.Dim)
	}
	if len(cfg.Target) != cfg.Dim {
		return nil, fmt.Errorf("target dim %d != agent dim %d", len(cfg.Target), cfg.Dim)
	}
	if cfg.Alpha <= 0 || cfg.Alpha >= 1 {
		cfg.Alpha = 0.3
	}
	if cfg.Depth <= 0 {
		cfg.Depth = 3
	}
	if cfg.Width <= 0 {
		cfg.Width = 5
	}
	if cfg.PerturbScale <= 0 {
		cfg.PerturbScale = 1.0
	}
	if cfg.FirewallThresh <= 0 {
		cfg.FirewallThresh = 5.0
	}

	maxBranches := cfg.Depth * cfg.Width

	agent := &CROMAgent{
		wm:        NewEMAWorldModel(cfg.Dim, cfg.Alpha),
		branches:  NewCROMBranchManager(cfg.Dim, maxBranches, cfg.PerturbScale, cfg.Seed),
		fw:        NewThresholdFirewall(cfg.FirewallThresh),
		dim:       cfg.Dim,
		target:    make([]float64, cfg.Dim),
		depth:     cfg.Depth,
		width:     cfg.Width,
		actionBuf: make([]float64, cfg.Dim),
	}
	copy(agent.target, cfg.Target)

	return agent, nil
}

// Step processes one observation and returns an action (1.1.7).
//
// Pipeline:
//  1. Update WorldModel with observation
//  2. Predict next state
//  3. Explore branches from prediction
//  4. Collapse to best branch (minimize free energy to target)
//  5. Compute action = best_branch_state - current_state
//  6. Firewall check
//  7. Sign decision (if firewall passes)
func (a *CROMAgent) Step(observation []float64) (*StepResult, error) {
	// 1. Update world model with observation
	a.wm.Update(observation)

	// 2. Predict next state
	prediction := a.wm.Predict(a.wm.GetState(), nil)

	// 3. Explore branches from prediction
	a.branches.Explore(prediction, a.depth, a.width)

	// 4. Collapse: select branch closest to target
	best, bestIdx := a.branches.Collapse(a.target)

	// 5. Compute action: direction from current state toward best branch
	for i := 0; i < a.dim; i++ {
		a.actionBuf[i] = best.State[i] - observation[i]
		// Clamp action magnitude
		if a.actionBuf[i] > 1.0 {
			a.actionBuf[i] = 1.0
		} else if a.actionBuf[i] < -1.0 {
			a.actionBuf[i] = -1.0
		}
	}

	// 6. Firewall check
	safe, confidence := a.fw.Check(a.actionBuf, observation)
	blocked := !safe

	// Update free energy
	a.freeEnergy = best.FreeEnergy + a.wm.Error()

	result := &StepResult{
		Action:       make([]float64, a.dim),
		FreeEnergy:   a.freeEnergy,
		BranchCount:  a.branches.BranchCount(),
		BestBranchID: bestIdx,
		Blocked:      blocked,
		PredError:    a.wm.Error(),
	}

	if blocked {
		// Return zero action when blocked
		for i := range result.Action {
			result.Action[i] = 0
		}
	} else {
		copy(result.Action, a.actionBuf)
		// Sign the decision
		_, _ = a.fw.Sign(float64sToBytes(a.actionBuf))
		_ = confidence
	}

	a.stepCount++
	return result, nil
}

// GetFreeEnergy returns the current free energy.
func (a *CROMAgent) GetFreeEnergy() float64 {
	return a.freeEnergy
}

// StepCount returns the number of steps executed.
func (a *CROMAgent) StepCount() int {
	return a.stepCount
}

// ============================================================================
// UTILITY
// ============================================================================

// float64sToBytes converts a float64 slice to bytes for signing.
// Uses unsafe-free approach via math.Float64bits.
func float64sToBytes(vals []float64) []byte {
	buf := make([]byte, len(vals)*8)
	for i, v := range vals {
		bits := math.Float64bits(v)
		off := i * 8
		buf[off+0] = byte(bits)
		buf[off+1] = byte(bits >> 8)
		buf[off+2] = byte(bits >> 16)
		buf[off+3] = byte(bits >> 24)
		buf[off+4] = byte(bits >> 32)
		buf[off+5] = byte(bits >> 40)
		buf[off+6] = byte(bits >> 48)
		buf[off+7] = byte(bits >> 56)
	}
	return buf
}
