package routing

import (
	"errors"
	"math"
	"sort"
	"sync"
)

// Constantes estipuladas para a PoC laboratorial de Roteamento Dinâmico.
const ContextDim = 896 // Alinhado com o Qwen2.5-0.5B-Instruct

// BrainNode representa um Cérebro Congelado e seu vetor de conhecimento principal.
type BrainNode struct {
	BrainID       string
	Centroid      []float32
	IsActive      bool
}

// HNSWRouter simula um orquestrador sub-simbólico.
// Mantém as referências aos arquivos FUSE mapeados para ponderar "Quem assume o volante".
type HNSWRouter struct {
	Mu     sync.RWMutex
	Brains []BrainNode
}

// NewRouter aloca o mecanismo de roteamento Multi-Brain em memória.
func NewRouter() *HNSWRouter {
	return &HNSWRouter{
		Brains: make([]BrainNode, 0),
	}
}

// RegisterBrain registra um novo cérebro especialista na malha semântica.
func (r *HNSWRouter) RegisterBrain(id string, contextVector []float32) error {
	r.Mu.Lock()
	defer r.Mu.Unlock()

	if len(contextVector) != ContextDim {
		return errors.New("vetor de contexto inválido para a topologia HNSW")
	}

	r.Brains = append(r.Brains, BrainNode{
		BrainID:  id,
		Centroid: contextVector,
		IsActive: true,
	})
	return nil
}

// DistanceResult encapsula os scores temporários.
type distanceResult struct {
	index int
	score float32
}

// GetTopKWeights avalia o Prompt (contextVector) contra todos os cérebros
// usando similaridade por cosseno. Retorna os Ponderadores (Pesos) ordenados.
func (r *HNSWRouter) GetTopKWeights(contextVector []float32, topK int) ([]float32, []int) {
	r.Mu.RLock()
	defer r.Mu.RUnlock()

	if len(r.Brains) == 0 {
		return nil, nil
	}

	totalTopK := topK
	if len(r.Brains) < topK {
		totalTopK = len(r.Brains)
	}

	results := make([]distanceResult, 0, len(r.Brains))

	// HNSW Traverse (Aproximação Linear PoC pela dimensionalidade pequena N < 5)
	for i, b := range r.Brains {
		if !b.IsActive {
			continue
		}
		sim := cosineSimilarity(contextVector, b.Centroid)
		results = append(results, distanceResult{index: i, score: sim})
	}

	// Ordena pelo maior score (Similaridade Positiva O(N log N))
	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})

	// Normalização Lógica (Softmax / Somatório Simples)
	// Como lidamos com simetria e pesos de deltas vetoriais, os topK vão somar 1.0.
	weights := make([]float32, totalTopK)
	indices := make([]int, totalTopK)

	var scoreSum float32
	for i := 0; i < totalTopK; i++ {
		score := results[i].score
		if score < 0 {
			score = 0 // Teto relu para penalizar direções reversas
		}
		scoreSum += score
	}

	for i := 0; i < totalTopK; i++ {
		if scoreSum > 0 {
			weights[i] = results[i].score / scoreSum
		} else {
			// Fallback uniforme caso o contexto seja oposto exato para todos
			weights[i] = 1.0 / float32(totalTopK)
		}
		indices[i] = results[i].index
	}

	return weights, indices
}

// Função nativa SIMD/CPU-bound de Cosseno (A.B / |A||B|)
func cosineSimilarity(a, b []float32) float32 {
	var dotProduct, normA, normB float32
	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / float32(math.Sqrt(float64(normA))*math.Sqrt(float64(normB)))
}
