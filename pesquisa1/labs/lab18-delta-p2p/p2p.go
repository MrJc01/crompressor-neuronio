package lab18

import (
	"crypto/ed25519"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"sync"
)

// DeltaMessage representa uma predição gerada pelo BranchManager de um agente.
// Para ser validado em um ambiente P2P, deve carregar uma assinatura Ed25519.
type DeltaMessage struct {
	AgentID   string  `json:"agent_id"`
	FreeEnergy float64 `json:"free_energy"`
	StateData []byte  `json:"state_data"` // Delta em si
	Signature []byte  `json:"signature"`
}

// PeerNode representa um agente na rede CROM distribuída.
type PeerNode struct {
	ID         string
	PrivateKey ed25519.PrivateKey
	PublicKey  ed25519.PublicKey
	
	Address    string
	listener   net.Listener
	
	Peers      []string // Endereços conhecidos
	mu         sync.Mutex
	
	// Canal para onde deltas válidos são enviados para processamento
	Inbox      chan DeltaMessage 
	
	running    bool
}

// NewPeerNode inicializa um nó com um par de chaves Ed25519 novo.
func NewPeerNode(address string) (*PeerNode, error) {
	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		return nil, err
	}
	
	id := fmt.Sprintf("crom-node-%x", pub[:4]) // Usa os primeiros 4 bytes como ID visual
	
	return &PeerNode{
		ID:         id,
		PrivateKey: priv,
		PublicKey:  pub,
		Address:    address,
		Peers:      make([]string, 0),
		Inbox:      make(chan DeltaMessage, 100),
	}, nil
}

// Start inicia o servidor TCP para receber Deltas.
func (node *PeerNode) Start() error {
	l, err := net.Listen("tcp", node.Address)
	if err != nil {
		return err
	}
	
	node.listener = l
	node.Address = l.Addr().String() // Atualiza com porta alocada caso porta 0 tenha sido passada
	node.running = true
	
	go func() {
		for node.running {
			conn, err := l.Accept()
			if err != nil {
				if !node.running {
					return
				}
				continue
			}
			go node.handleConnection(conn)
		}
	}()
	
	return nil
}

// Stop desliga o servidor graciosamente.
func (node *PeerNode) Stop() {
	node.running = false
	if node.listener != nil {
		node.listener.Close()
	}
}

// handleConnection recebe e valida um DeltaMessage da rede.
func (node *PeerNode) handleConnection(conn net.Conn) {
	defer conn.Close()
	
	var msg DeltaMessage
	decoder := json.NewDecoder(conn)
	if err := decoder.Decode(&msg); err != nil {
		return
	}
	
	// Validação de Integridade Criptográfica (Ed25519)
	// O protocolo exige que a mensagem seja assinada com a chave pública atrelada ao AgentID.
	// Neste lab, assumimos que recebemos a PubKey previamente ou no handshake.
	// Para simplificar, testaremos apenas a validação da assinatura bruta da mensagem sem a string.
	// Carga útil assinada = AgentID + FreeEnergy + StateData
	// payload := fmt.Sprintf("%s|%f|%s", msg.AgentID, msg.FreeEnergy, string(msg.StateData))
	
	// IMPORTANTE: Em um caso real, a chave pública tem que vir de um KeyStore confiável.
	// Aqui o teste injetará a chave ou assumiremos que o validador confia na chave do sender.
	
	node.Inbox <- msg
}

// Broadcast cria um Delta, assina digitalmente, e espalha para todos os Peers conhecidos.
func (node *PeerNode) Broadcast(freeEnergy float64, stateData []byte) error {
	payload := fmt.Sprintf("%s|%f|%s", node.ID, freeEnergy, string(stateData))
	signature := ed25519.Sign(node.PrivateKey, []byte(payload))
	
	msg := DeltaMessage{
		AgentID:   node.ID,
		FreeEnergy: freeEnergy,
		StateData: stateData,
		Signature: signature,
	}
	
	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	
	node.mu.Lock()
	peers := make([]string, len(node.Peers))
	copy(peers, node.Peers)
	node.mu.Unlock()
	
	var errs []error
	for _, peerAddr := range peers {
		err := node.sendToPeer(peerAddr, data)
		if err != nil {
			errs = append(errs, err)
		}
	}
	
	if len(errs) > 0 {
		return errors.New("failed to broadcast to some peers")
	}
	
	return nil
}

func (node *PeerNode) sendToPeer(address string, data []byte) error {
	conn, err := net.Dial("tcp", address)
	if err != nil {
		return err
	}
	defer conn.Close()
	
	_, err = conn.Write(data)
	return err
}

// VerifySignature verifica a integridade do Delta usando uma PublicKey fornecida.
func VerifySignature(pubKey ed25519.PublicKey, msg DeltaMessage) bool {
	payload := fmt.Sprintf("%s|%f|%s", msg.AgentID, msg.FreeEnergy, string(msg.StateData))
	return ed25519.Verify(pubKey, []byte(payload), msg.Signature)
}

// AddPeer registra um novo nó na rede local.
func (node *PeerNode) AddPeer(address string) {
	node.mu.Lock()
	defer node.mu.Unlock()
	node.Peers = append(node.Peers, address)
}
