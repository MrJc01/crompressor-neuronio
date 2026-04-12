package tensor

import (
	"errors"
)

// ComposeDeltas representa a arquitetura Plástica do ecossistema.
// Ele recebe N deltas provenientes de diferentes "Brains" e as matrizes de peso (HNSW).
// Ele funde essas variações quantizadas no array de bytes linear originais (Frozen Neuron).
func ComposeDeltas(original []byte, deltas [][]byte, weights []float32) ([]byte, error) {
	if len(deltas) != len(weights) {
		return nil, errors.New("o número de matrizes delta deve ser matematicamente condizente com os pesos do roteador")
	}

	if len(deltas) == 0 {
		return original, nil
	}

	size := len(original)
	for _, d := range deltas {
		if len(d) != size {
			return nil, errors.New("vetores delta apresentam assimetria espacial contra o chunk original GGUF")
		}
	}

	result := make([]byte, size)

	// Aritmética Dinâmica Vector Quantization + XOR composition
	// No conceito de Cérebro Múltiplo, em vez de um XOR destrutivo simples puro (A^B),
	// processamos o "Delta Ponderado" de forma associativa flutuante.
	for i := 0; i < size; i++ {
		var weightedDelta float32

		// 1. Extração do Ponderador
		for brainIdx := 0; brainIdx < len(deltas); brainIdx++ {
			// O Valor do Delta é a Diferencia Estrita = (deltaByte - originalByte)
			// Porque nossos tensores falsos estão gravados como XOR Puros até o momento.
			// Na vida real este loop itera floats nas malhas VQ de K-médias.
			deltaVal := float32(int(deltas[brainIdx][i]) - int(original[i])) 
			weightedDelta += deltaVal * weights[brainIdx]
		}

		// 2. Aplicação Reversa
		computedByte := float32(original[i]) + weightedDelta

		// 3. Clamping do Espectro de Memória para não corromper bits GGUF
		if computedByte > 255.0 {
			computedByte = 255.0
		} else if computedByte < 0 {
			computedByte = 0
		}

		result[i] = byte(computedByte)
	}

	return result, nil
}
