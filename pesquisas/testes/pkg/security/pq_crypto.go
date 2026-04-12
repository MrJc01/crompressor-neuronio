package security

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/ed25519"
	"crypto/rand"
	"errors"
	"fmt"
	"io"
)

// CromPayload representa o envelope seguro de um Tensor Delta.
type CromPayload struct {
	Signature []byte // Assinatura Digital (Placeholder para Dilithium/Ed25519)
	Nonce     []byte // Nonce para encriptação
	Ciphertext []byte // Dados do Delta XOR encriptados
}

// GenerateCromKeys gera um par de chaves para assinatura soberana dos neurônios.
func GenerateCromKeys() (ed25519.PublicKey, ed25519.PrivateKey, error) {
	return ed25519.GenerateKey(rand.Reader)
}

// SealDelta assina e encripta um Tensor Delta usando AES-GCM e Ed25519.
// Na Fase 4 real, o Ed25519 seria substituído por Dilithium para proteção PQC.
func SealDelta(delta []byte, privKey ed25519.PrivateKey, secretKey []byte) (*CromPayload, error) {
	if len(secretKey) != 32 {
		return nil, errors.New("a chave secreta deve ter 32 bytes (AES-256)")
	}

	// 1. Assinar o Delta original (Soberania do Criador)
	signature := ed25519.Sign(privKey, delta)

	// 2. Encriptar o Delta (Proteção contra espionagem no P2P)
	block, err := aes.NewCipher(secretKey)
	if err != nil {
		return nil, err
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}

	nonce := make([]byte, gcm.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, err
	}

	ciphertext := gcm.Seal(nil, nonce, delta, nil)

	return &CromPayload{
		Signature:  signature,
		Nonce:      nonce,
		Ciphertext: ciphertext,
	}, nil
}

// OpenDelta descriptografa e verifica a assinatura de um envelope recebido via rede.
func OpenDelta(payload *CromPayload, pubKey ed25519.PublicKey, secretKey []byte) ([]byte, error) {
	if len(secretKey) != 32 {
		return nil, errors.New("a chave secreta deve ter 32 bytes")
	}

	// 1. Descriptografar
	block, err := aes.NewCipher(secretKey)
	if err != nil {
		return nil, err
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}

	delta, err := gcm.Open(nil, payload.Nonce, payload.Ciphertext, nil)
	if err != nil {
		return nil, fmt.Errorf("falha na integridade da encriptação: %v", err)
	}

	// 2. Verificar Assinatura (Validar que o neurônio veio de fonte confiável)
	if !ed25519.Verify(pubKey, delta, payload.Signature) {
		return nil, errors.New("assinatura inválida: o neurônio pode ter sido envenenado ou alterado")
	}

	return delta, nil
}
