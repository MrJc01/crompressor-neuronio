package lab15

import (
	"strings"
	"testing"
)

func TestEnsembleDetector(t *testing.T) {
	context := "A compressão neural baseada em K-Means reduz a memória em mais de noventa por cento e mantém a divergência KL estritamente intacta."

	// Configuração do Ensemble
	ensemble := &EnsembleDetector{
		V1: &NgramDetector{N: 2},           // Precisa de 2 palavras exatas para não alarmar
		V2: &JaccardDetector{Threshold: 0.15}, // Threshold baixo, focado em penalizar invenções totais
		V3: &SemanticDetector{},              // Mockado por caso de teste
	}

	tests := []struct {
		name         string
		sentence     string
		v3Says       bool // True = V3 acha que é alucinação
		wantEnsemble bool // True = Esperamos que o Ensemble bloqueie
		reason       string
	}{
		{
			name:         "Caso 1: Real - V3 Aprova (Recall 100% path)",
			sentence:     "K-Means reduz a memória mantendo divergência KL intacta",
			v3Says:       false, // V3 tem recall de 100%, então se for seguro, ele diz que é seguro.
			wantEnsemble: false,
			reason:       "Se V3 não apita alucinação, confiamos cegamente que é seguro.",
		},
		{
			name:         "Caso 2: Alucinação Tota - V1 Intercepta",
			sentence:     "O modelo GPT-4 custa milhares de dólares por minuto.",
			v3Says:       true, // V3 apita alucinação
			wantEnsemble: true, // V1 também vai apitar, pq não tem overlap de Ngrams
			reason:       "Se V3 apita, e V1 apita (zero ngrams originais), o ensemble bloqueia instantaneamente.",
		},
		{
			name:         "Caso 3: Falso Positivo do V3 - V1 Intercepta Seguro",
			sentence:     "reduz a memória em mais",
			v3Says:       true,  // O V3 é hiper-sensível, achou que é alucinação porque tá curto.
			wantEnsemble: false, // O V1 acha os exatos ngrams! V1 diz q é seguro. V2 entra pra desempatar. Jaccard é alto!
			reason:       "V3 apitou, mas V1 perdoou (encontrou ngram). V2 foi chamado e o Jaccard permitiu.",
		},
		{
			name:         "Caso 4: Ataque de Paráfrase Distante",
			sentence:     "A técnica matemática encolhe o tamanho brutalmente.",
			v3Says:       true, // V3 apita alucinação
			// V1 não vai achar overlap. Então V1 APITA! 
			wantEnsemble: true,
			reason:       "V3 apita e V1 apita (nenhum ngram de 2). Bloqueado.",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Mockando o comportamento do ML pesado
			ensemble.V3.PredictFunc = func(ctx, sen string) bool {
				// Verifica se os inputs bateram
				if !strings.EqualFold(ctx, context) || !strings.EqualFold(sen, tt.sentence) {
					t.Fatalf("V3 mock recebeu inputs inválidos")
				}
				return tt.v3Says
			}

			got := ensemble.Check(context, tt.sentence)
			if got != tt.wantEnsemble {
				t.Errorf("Ensemble falhou. \nSentença: %q\nV3 disse: %v\nEsperado: %v, Recebido: %v\nMotivo: %s",
					tt.sentence, tt.v3Says, tt.wantEnsemble, got, tt.reason)
			}
		})
	}
}
