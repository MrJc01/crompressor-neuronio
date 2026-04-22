// Firewall implementation with threshold-based anomaly detection and Ed25519 signing.
//
// Item 1.1.5: Implementar Firewall com threshold de erro
// Item 1.1.6: Integrar Ed25519 para assinar cada decisão
//
// Critério de Sucesso:
//   - ≥70% de alucinações bloqueadas
//   - Sign <50μs, Verify <200μs
package lab13

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"math"
)

// ThresholdFirewall blocks predictions whose error exceeds a threshold
// and signs approved decisions with Ed25519.
type ThresholdFirewall struct {
	threshold float64

	// Ed25519 keypair (generated on init)
	privKey ed25519.PrivateKey
	pubKey  ed25519.PublicKey

	// Statistics
	stats FirewallStats
}

// NewThresholdFirewall creates a firewall with the given error threshold.
// Generates a fresh Ed25519 keypair for signing.
func NewThresholdFirewall(threshold float64) *ThresholdFirewall {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	return &ThresholdFirewall{
		threshold: threshold,
		privKey:   priv,
		pubKey:    pub,
	}
}

// Check determines if a prediction is safe by comparing its magnitude
// against the reference and the configured threshold.
//
// The confidence is calculated as: 1.0 - (error / threshold)
// Returns safe=true if the error is within bounds.
func (f *ThresholdFirewall) Check(prediction, reference []float64) (safe bool, confidence float64) {
	f.stats.TotalChecks++

	// Calculate error magnitude (Euclidean distance)
	errMag := 0.0
	dim := len(prediction)
	if len(reference) < dim {
		dim = len(reference)
	}
	for i := 0; i < dim; i++ {
		d := prediction[i] - reference[i]
		errMag += d * d
	}
	errMag = math.Sqrt(errMag)

	if errMag > f.threshold {
		f.stats.BlockedCount++
		return false, 0.0
	}

	f.stats.PassedCount++
	confidence = 1.0 - (errMag / f.threshold)
	if confidence < 0 {
		confidence = 0
	}
	return true, confidence
}

// Sign signs the given data using Ed25519.
// The data is first hashed with SHA-256 before signing.
func (f *ThresholdFirewall) Sign(data []byte) ([]byte, error) {
	hash := sha256.Sum256(data)
	sig := ed25519.Sign(f.privKey, hash[:])
	f.stats.SignedCount++
	return sig, nil
}

// Verify verifies an Ed25519 signature against the data.
func (f *ThresholdFirewall) Verify(data, signature []byte) bool {
	hash := sha256.Sum256(data)
	valid := ed25519.Verify(f.pubKey, hash[:], signature)
	f.stats.VerifiedCount++
	return valid
}

// Stats returns the current firewall statistics.
func (f *ThresholdFirewall) Stats() FirewallStats {
	return f.stats
}

// GetPublicKey returns the Ed25519 public key.
func (f *ThresholdFirewall) GetPublicKey() ed25519.PublicKey {
	return f.pubKey
}
