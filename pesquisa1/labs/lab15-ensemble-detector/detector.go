package lab15

import (
	"strings"
)

// Detector define a interface de um módulo de detecção de alucinações.
type Detector interface {
	// Check avalia uma sentença contra um contexto.
	// Retorna true se a sentença for detectada como ALUCINAÇÃO.
	Check(context, sentence string) bool
}

// ============================================================================
// V1: N-Gram Overlap Detector (Alta Precision, Baixo Recall)
// ============================================================================

type NgramDetector struct {
	N int
}

func (d *NgramDetector) Check(context, sentence string) bool {
	ctxWords := strings.Fields(strings.ToLower(context))
	senWords := strings.Fields(strings.ToLower(sentence))

	if len(senWords) < d.N {
		return false // Impossível formar ngram
	}

	// Cria mapa de ngrams do contexto
	ctxNgrams := make(map[string]bool)
	for i := 0; i <= len(ctxWords)-d.N; i++ {
		gram := strings.Join(ctxWords[i:i+d.N], " ")
		ctxNgrams[gram] = true
	}

	// Verifica se PELA MENOS UM ngram da sentença existe no contexto
	// Se nenhum ngram existir, é muito provável que seja alucinação
	hasOverlap := false
	for i := 0; i <= len(senWords)-d.N; i++ {
		gram := strings.Join(senWords[i:i+d.N], " ")
		if ctxNgrams[gram] {
			hasOverlap = true
			break
		}
	}

	// Alucinação = ausência de overlap de n-gramas
	return !hasOverlap
}

// ============================================================================
// V2: Jaccard Similarity (Equilíbrio / Fallback)
// ============================================================================

type JaccardDetector struct {
	Threshold float64
}

func (d *JaccardDetector) Check(context, sentence string) bool {
	ctxWords := strings.Fields(strings.ToLower(context))
	senWords := strings.Fields(strings.ToLower(sentence))

	setA := make(map[string]bool)
	for _, w := range ctxWords {
		setA[w] = true
	}

	setB := make(map[string]bool)
	for _, w := range senWords {
		setB[w] = true
	}

	intersection := 0
	for w := range setB {
		if setA[w] {
			intersection++
		}
	}

	union := len(setA) + len(setB) - intersection
	if union == 0 {
		return true
	}

	score := float64(intersection) / float64(union)
	
	// Alucinação = similaridade abaixo do threshold
	return score < d.Threshold
}

// ============================================================================
// V3: Semantic/Vector Detector (Mocked para Go puro) (100% Recall)
// ============================================================================
// Em produção, isso bateria num endpoint SBERT ou buscaria do KV Cache quantizado.

type SemanticDetector struct {
	// Mock function para injetar comportamento de ML real nos testes
	PredictFunc func(context, sentence string) bool
}

func (d *SemanticDetector) Check(context, sentence string) bool {
	if d.PredictFunc != nil {
		return d.PredictFunc(context, sentence)
	}
	return false
}

// ============================================================================
// ENSEMBLE DETECTOR
// ============================================================================

type EnsembleDetector struct {
	V1 *NgramDetector
	V2 *JaccardDetector
	V3 *SemanticDetector
}

func (e *EnsembleDetector) Check(context, sentence string) bool {
	// 1. V3: O Filtro de 100% Recall
	// Se o V3 não apontar alucinação, confiamos cegamente.
	isHallucinationV3 := e.V3.Check(context, sentence)
	if !isHallucinationV3 {
		return false // Seguro
	}

	// 2. V1: A Lâmina de 100% Precision
	// Se o V1 apontar alucinação (falha em achar n-gramas exatos), confiamos cegamente.
	isHallucinationV1 := e.V1.Check(context, sentence)
	if isHallucinationV1 {
		return true // Alucinação confirmada
	}

	// 3. V2: O Desempate
	// Chegamos aqui se V3 diz que é alucinação (falso positivo potencial) e V1 diz que é seguro.
	// O V2 atua como juiz usando uma heurística intermediária.
	return e.V2.Check(context, sentence)
}
