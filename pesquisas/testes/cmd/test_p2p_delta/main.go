package main

import (
	"bytes"
	"fmt"
	"log"

	"github.com/MrJc01/crompressor-neuronio/pesquisas/testes/pkg/security"
)

func main() {
	fmt.Println("╔═══════════════════════════════════════════════════════╗")
	echo("║   🧬 P2P SECURITY TEST - SOVEREIGN NEURONS          ║")
	fmt.Println("╚═══════════════════════════════════════════════════════╝")

	// 1. Setup: Chaves de Identidade Soberana e Chave de Sessão P2P
	fmt.Println("⏳ Gerando Identidade Soberana (PQC-Ready keys)...")
	pub, priv, err := security.GenerateCromKeys()
	if err != nil {
		log.Fatal(err)
	}

	sessionKey := make([]byte, 32)
	for i := range sessionKey {
		sessionKey[i] = byte(i) // Chave de simulação AES-256
	}

	// 2. Criação do Delta (Neurônio Matemático)
	originalDelta := []byte("PESOS_NEURAIS_DELTA_QUANTIZADOS_Q4_K_M_V1.0")
	fmt.Printf("📦 Delta Original: [%s]\n", string(originalDelta))

	// 3. Selagem (Assinatura + Encriptação)
	fmt.Println("🔒 Selando Neurônio para tráfego em rede hostil...")
	payload, err := security.SealDelta(originalDelta, priv, sessionKey)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("   -> Tamanho do Ciphertext: %d bytes\n", len(payload.Ciphertext))
	fmt.Printf("   -> Hash Assinatura Ed25519: %x...\n", payload.Signature[:16])

	// 4. Teste de Sucesso (Abertura Íntegra)
	fmt.Println("\n✅ Simulando Recebimento no Nó Remoto (Integridade OK)...")
	recoveredDelta, err := security.OpenDelta(payload, pub, sessionKey)
	if err != nil {
		log.Fatalf("❌ Falha crítica: %v", err)
	}

	if bytes.Equal(originalDelta, recoveredDelta) {
		fmt.Printf("🎉 Sucesso! Delta recuperado perfeitamente: [%s]\n", string(recoveredDelta))
	}

	// 5. Teste de Resistência (Simulando Ataque de Envenenamento/Manipulação)
	fmt.Println("\n🛡️  Simulando Ataque de Envenenamento (Bit Flip no Ciphertext)...")
	payload.Ciphertext[5] ^= 0xFF // Alterando um byte no meio da transmissão

	_, err = security.OpenDelta(payload, pub, sessionKey)
	if err != nil {
		fmt.Printf("🛡️  SISTEMA REJEITOU PAYLOAD: %v\n", err)
		fmt.Println("    [OK] O ataque de envenenamento foi detectado pelo motor AEAD/Assinatura.")
	} else {
		fmt.Println("❌ ERRO: O sistema aceitou um payload corrompido! Falha na criptografia.")
	}
}

func echo(s string) {
	fmt.Println(s)
}
