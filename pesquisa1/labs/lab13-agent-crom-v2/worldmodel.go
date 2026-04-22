// WorldModel implementation using Exponential Moving Average (EMA).
//
// Item 1.1.2: Implementar WorldModel com EMA
// Critério de Sucesso: Erro <5% em sequência sintética
package lab13

import "math"

const defaultErrBufSize = 32

// EMAWorldModel implements WorldModel using Exponential Moving Average.
//
// The model maintains:
//   - state: current position estimate (EMA-smoothed)
//   - velocity: estimated rate of change (EMA-smoothed)
//   - predBuf: preallocated buffer for predictions (zero-alloc)
//
// Prediction: next_state = state + velocity + action
// Update:     state = α·observation + (1-α)·state
//
//	velocity = α·(observation - old_state) + (1-α)·velocity
type EMAWorldModel struct {
	state    []float64 // current state estimate
	velocity []float64 // estimated velocity (change per step)
	alpha    float64   // smoothing factor (0-1), higher = more responsive
	dim      int

	// Error tracking (circular buffer, no allocations after init)
	errBuf   []float64
	errIdx   int
	errCount int
	errSum   float64

	// Preallocated buffers
	predBuf []float64
}

// NewEMAWorldModel creates a new EMA-based world model.
//
// Parameters:
//   - dim: dimensionality of the state space
//   - alpha: EMA smoothing factor (0.0-1.0), recommended 0.3
func NewEMAWorldModel(dim int, alpha float64) *EMAWorldModel {
	if alpha <= 0 || alpha >= 1 {
		alpha = 0.3
	}
	return &EMAWorldModel{
		state:    make([]float64, dim),
		velocity: make([]float64, dim),
		alpha:    alpha,
		dim:      dim,
		errBuf:   make([]float64, defaultErrBufSize),
		predBuf:  make([]float64, dim),
	}
}

// Predict estimates the next state: next = state + velocity + action.
//
// Returns internal buffer — do not modify. Valid until next Predict call.
// This is allocation-free on the hot path.
func (w *EMAWorldModel) Predict(state, action []float64) []float64 {
	for i := 0; i < w.dim; i++ {
		w.predBuf[i] = w.state[i] + w.velocity[i]
		if action != nil && i < len(action) {
			w.predBuf[i] += action[i]
		}
	}
	return w.predBuf
}

// Update corrects the model with a real observation using EMA smoothing.
//
// Steps:
//  1. Compute RMSE between observation and current state
//  2. Update velocity estimate with EMA
//  3. Update state estimate with EMA
//  4. Record error in circular buffer
func (w *EMAWorldModel) Update(observation []float64) {
	errVal := 0.0
	for i := 0; i < w.dim && i < len(observation); i++ {
		diff := observation[i] - w.state[i]
		errVal += diff * diff

		// Update velocity estimate (how fast state is changing)
		w.velocity[i] = w.alpha*diff + (1-w.alpha)*w.velocity[i]

		// Update state with EMA
		w.state[i] = w.alpha*observation[i] + (1-w.alpha)*w.state[i]
	}
	errVal = math.Sqrt(errVal / float64(w.dim))

	// Track error in circular buffer (fixed-size, no allocation)
	if w.errCount >= len(w.errBuf) {
		w.errSum -= w.errBuf[w.errIdx]
	} else {
		w.errCount++
	}
	w.errBuf[w.errIdx] = errVal
	w.errSum += errVal
	w.errIdx = (w.errIdx + 1) % len(w.errBuf)
}

// Error returns the average recent prediction error (RMSE).
func (w *EMAWorldModel) Error() float64 {
	if w.errCount == 0 {
		return 0
	}
	return w.errSum / float64(w.errCount)
}

// GetState returns the current estimated state.
// WARNING: Returns internal buffer, do not modify.
func (w *EMAWorldModel) GetState() []float64 {
	return w.state
}

// SetState sets the initial state of the model.
func (w *EMAWorldModel) SetState(state []float64) {
	copy(w.state, state)
}
