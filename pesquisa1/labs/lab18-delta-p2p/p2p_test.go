package lab18

import (
	"testing"
	"time"
)

func TestP2PDeltaExchange(t *testing.T) {
	// 1. Instanciar dois nós na rede local usando portas aleatórias
	nodeA, err := NewPeerNode("127.0.0.1:0")
	if err != nil {
		t.Fatalf("Failed to create nodeA: %v", err)
	}
	
	nodeB, err := NewPeerNode("127.0.0.1:0")
	if err != nil {
		t.Fatalf("Failed to create nodeB: %v", err)
	}

	// 2. Iniciar servidores
	if err := nodeA.Start(); err != nil {
		t.Fatalf("nodeA failed to start: %v", err)
	}
	defer nodeA.Stop()

	if err := nodeB.Start(); err != nil {
		t.Fatalf("nodeB failed to start: %v", err)
	}
	defer nodeB.Stop()

	// 3. Conectar os nós (Simulando o Discovery)
	nodeA.AddPeer(nodeB.Address)

	// 4. Node A transmite um "pensamento" (Delta)
	// No CROM, o agente com menor Free Energy lidera o consenso.
	fakeState := []byte("branch_id_42_action_up")
	lowFreeEnergy := 0.15 
	
	err = nodeA.Broadcast(lowFreeEnergy, fakeState)
	if err != nil {
		t.Fatalf("Broadcast failed: %v", err)
	}

	// 5. Node B deve receber e validar a mensagem
	select {
	case msg := <-nodeB.Inbox:
		// Verificar se os dados bateram
		if msg.AgentID != nodeA.ID {
			t.Errorf("Expected AgentID %s, got %s", nodeA.ID, msg.AgentID)
		}
		if msg.FreeEnergy != lowFreeEnergy {
			t.Errorf("Expected FreeEnergy %f, got %f", lowFreeEnergy, msg.FreeEnergy)
		}

		// 6. Validar Assinatura Ed25519 (Segurança da Rede)
		// No mundo real, Node B busca a publicKey de Node A na tabela DHT. Aqui usamos direto.
		isValid := VerifySignature(nodeA.PublicKey, msg)
		if !isValid {
			t.Errorf("Signature verification failed! Delta rejected.")
		} else {
			t.Logf("✅ Success! Delta received and verified. Node %s sent F=%.2f", msg.AgentID, msg.FreeEnergy)
		}

	case <-time.After(2 * time.Second):
		t.Fatal("Timeout waiting for DeltaMessage on Node B")
	}
}

func TestP2PFakeSignatureRejection(t *testing.T) {
	nodeA, _ := NewPeerNode("127.0.0.1:0")
	// nodeB := malicioso, ignorado pois geramos a mensagem falsa manualmente
	
	// Malicious B generates message but signs with his own key, 
	// while pretending to be A!
	fakePayload := []byte("hack_the_system")
	msg := DeltaMessage{
		AgentID:   nodeA.ID, // Spoofing!
		FreeEnergy: 0.0,
		StateData: fakePayload,
	}
	
	// Signs the spoofed payload with B's private key
	// B signs it
	msg.Signature = []byte("fake_signature_bytes_from_hacker") 
	
	// When another node tries to verify assuming the message came from A (using A's public key)
	isValid := VerifySignature(nodeA.PublicKey, msg)
	
	if isValid {
		t.Errorf("Security Flaw! Spoofed signature was accepted.")
	} else {
		t.Logf("✅ Success! Spoofed signature rejected by Ed25519 firewall.")
	}
}
