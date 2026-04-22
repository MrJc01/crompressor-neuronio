package main

import (
	"fmt"
	"math/rand"
	"syscall/js"
)

// ============================================================================
// Lab23: WASM Edge Deployment do Motor CROM
// ============================================================================
// Este código compila para WASM e expõe o CROM Agent para rodar dentro
// do browser (JavaScript DOM) alcançando as mesmas latências sub-microssegundo.

var worldModel []float64

// cromStep é a função exportada para o JavaScript.
// Simula 1 "Step" de Active Inference.
func cromStep(this js.Value, p []js.Value) interface{} {
	if len(p) < 1 {
		return "Erro: Falta observation input"
	}
	
	obsInput := p[0].Float()
	
	// Motor CROM super-simplificado para o benchmark WASM:
	// EMA (Exponential Moving Average) World Model Update
	alpha := 0.3
	prediction := 0.0
	if len(worldModel) > 0 {
		prediction = worldModel[len(worldModel)-1]
	}
	
	newPrediction := prediction + alpha*(obsInput-prediction)
	worldModel = append(worldModel, newPrediction)
	
	// Simulação do BranchManager (gerando caminhos delta)
	freeEnergy := (obsInput - newPrediction) * (obsInput - newPrediction)
	
	action := rand.Float64() // Simula "collapse" da decisão
	
	// Retornamos os resultados estruturados para o JavaScript
	return js.ValueOf(map[string]interface{}{
		"prediction": newPrediction,
		"free_energy": freeEnergy,
		"action": action,
	})
}

// cromInit zera a memória do modelo, pronto para novos estímulos.
func cromInit(this js.Value, p []js.Value) interface{} {
	worldModel = make([]float64, 0, 1000)
	fmt.Println("WASM: CROM Motor inicializado. Memória limpa.")
	return js.ValueOf(true)
}

func main() {
	// A rotina main no WASM fica "escutando" indefinidamente num channel
	// enquanto as funções exportadas podem ser chamadas via `syscall/js`.
	c := make(chan struct{}, 0)

	fmt.Println("WASM: Crompressor-Neuronio Neural Engine loaded.")

	// Exportando funções nativas do Go para o escopo global (window) do JS
	js.Global().Set("cromStep", js.FuncOf(cromStep))
	js.Global().Set("cromInit", js.FuncOf(cromInit))

	<-c
}
