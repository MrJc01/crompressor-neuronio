# 🔐 Segurança & Soberania

> *"Compartilhe inteligência, nunca o modelo."*

---

## Modelo de Ameaça

| Ameaça | Risco | Mitigação |
|:---|:---|:---|
| Extração do modelo via delta | Alto | Delta é inútil sem brain.crom compatível |
| Adulteração do delta | Alto | Assinatura Dilithium pós-quântica |
| Interceptação P2P | Médio | ChaCha20-Poly1305 (via crompressor-security) |
| Replay de deltas antigos | Médio | Nonce + timestamp no header do delta |
| Sybil attack na rede P2P | Baixo | Rate limiting + Silent Drop |
| Corrupção do brain.crom | Baixo | Merkle Tree verifica cada chunk |
| Quantum computing futuro | Médio | Dilithium + ChaCha20 são PQ-resistentes |

---

## Camada 1: Integridade via Merkle Tree

Cada chunk do brain.crom possui um hash na Merkle Tree:

```
                    MerkleRoot
                    /         \
               H(01)          H(23)
              /     \        /     \
          H(0)    H(1)   H(2)    H(3)
           |       |       |       |
        chunk0  chunk1  chunk2  chunk3
```

### Verificação Parcial (Semi-Fixo)
Quando chunks são atualizados (Vertente 2), apenas o ramo afetado é recalculado:

```
Atualizar chunk2:
  1. Recalcular H(2) = hash(chunk2_novo)
  2. Recalcular H(23) = hash(H(2) || H(3))
  3. Recalcular MerkleRoot = hash(H(01) || H(23))
  
Custo: O(log N) em vez de O(N)
```

---

## Camada 2: Assinatura Pós-Quântica

Usando as primitivas do crompressor-security:

```go
// Assinar um neurônio ou delta
type SignedArtifact struct {
    Data      []byte          // brain.crom ou delta.bin
    Signature []byte          // Dilithium signature (2420 bytes)
    PublicKey []byte          // Dilithium public key (1312 bytes)
    Timestamp int64           // Unix timestamp
    Nonce     [24]byte        // Anti-replay
}

// Verificação
func Verify(artifact SignedArtifact) bool {
    return dilithium.Verify(
        artifact.PublicKey,
        artifact.Data,
        artifact.Signature,
    )
}
```

### Por Que Dilithium?
- Resistente a computação quântica (NIST PQC Round 3)
- Tamanho da assinatura: 2420 bytes (aceitável para deltas de KB)
- Verificação: ~0.5ms (viável em tempo real)

---

## Camada 3: Criptografia em-Flight

Para trocas P2P, o delta é criptografado com ChaCha20-Poly1305:

```
Remetente:
  delta.bin → ChaCha20-Poly1305(key, nonce) → delta.enc

Transmissão P2P (Kademlia/LibP2P):
  delta.enc → rede (criptografado, ilegível)

Receptor:
  delta.enc → ChaCha20-Poly1305_decrypt(key, nonce) → delta.bin
  → Verificar Dilithium signature
  → Aplicar sobre brain.crom local
```

### Propriedades
- **O(1) memória:** Streaming cipher (não precisa do arquivo inteiro em RAM)
- **AEAD:** Authenticated Encryption with Associated Data
- **PQ-safe:** ChaCha20 é considerado seguro contra quantum

---

## Camada 4: Soberania do Modelo

### O Princípio
```
O modelo (brain.crom) NUNCA sai do dispositivo do proprietário.
Apenas tensores delta são compartilhados.
Um delta sem o brain.crom correspondente é dados sem sentido.
```

### Prova Matemática
```
Dado:
  brain.crom = B (dados do modelo comprimido, ~2 GB)
  delta.bin  = D (diferencial, ~100 KB)
  
Para reconstruir output:
  output = F(B ⊕ D)
  
Sem B, o atacante tem apenas D.
D é estatisticamente indistinguível de ruído aleatório
(porque é XOR de B com uma variante de B, e B é desconhecido).

Informação teórica de Shannon:
  H(D | sem B) ≈ H(ruído aleatório) = máxima entropia
  → Nenhuma informação útil pode ser extraída
```

### Zero-Knowledge de Capacidade
Um nó pode provar que possui uma capacidade (ex: "eu sei medicina") sem revelar o modelo:

```
1. Desafiante envia prompt médico P
2. Nó gera resposta R = F(brain_med.crom ⊕ delta)
3. Nó envia hash(R) como compromisso
4. Desafiante revela resposta esperada E
5. Se hash(R) corresponde a qualidade médica → prova de capacidade
6. Em nenhum momento brain_med.crom foi exposto
```

---

## Checklist de Segurança

- [ ] Merkle Tree verificada antes de cada montagem FUSE
- [ ] Dilithium signature obrigatória para deltas P2P
- [ ] ChaCha20-Poly1305 para toda comunicação P2P
- [ ] Nonce + timestamp para anti-replay
- [ ] Silent Drop ativo para conexões desconhecidas
- [ ] Rate limiting por IP (via crompressor-security)
- [ ] Logs de integridade em `pesquisas/relatorios/security_log.json`

---

> **Próximo:** [08 — Benchmarks Esperados](08-BENCHMARKS-ESPERADOS.md)
