"""
CROM P2P Network v3 — Criptografia Ed25519 REAL
================================================
Usa a biblioteca `cryptography` para assinar e verificar pacotes com
chaves Ed25519 reais. Nenhuma string fake. Nenhum if hardcoded.

A validação criptográfica funciona pela MATEMÁTICA, não por comparação de string.
"""
import time
import threading
import queue
import secrets
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# Barramento de rede (simula broadcast UDP local)
network_bus = []


class P2PNode(threading.Thread):
    def __init__(self, node_id, is_malicious=False):
        super().__init__(daemon=True)
        self.node_id = node_id
        self.is_malicious = is_malicious
        self.inbox = queue.Queue()
        
        # Gerar par de chaves Ed25519 REAL
        self.private_key = Ed25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()
        self.pub_bytes = self.public_key.public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw
        )
        
        network_bus.append(self)
    
    def sign_message(self, payload: bytes) -> bytes:
        """Assinatura Ed25519 real — 64 bytes de curva elíptica."""
        return self.private_key.sign(payload)
    
    def verify_signature(self, payload: bytes, signature: bytes, pub_key) -> bool:
        """Verificação criptográfica real. Falha = InvalidSignature exception."""
        try:
            pub_key.verify(signature, payload)
            return True
        except InvalidSignature:
            return False
    
    def broadcast(self, free_energy_delta: float):
        """Envia um delta de estado neural assinado para toda a rede."""
        nonce = secrets.token_bytes(16)
        timestamp = time.time()
        
        # Payload: dados + nonce + timestamp
        payload = f"TENSOR_DELTA_F={free_energy_delta:.6f}|NONCE={nonce.hex()}|TS={timestamp}".encode()
        
        if self.is_malicious:
            # O hacker tenta forjar uma assinatura — gera bytes aleatórios
            fake_sig = secrets.token_bytes(64)
            fake_key = Ed25519PrivateKey.generate().public_key()  # Chave que não corresponde
            print(f"{Colors.FAIL}  [NÓ {self.node_id}] 💀 HACKER: Injetando pacote com assinatura FORJADA{Colors.ENDC}")
            
            message = {
                'sender_id': self.node_id,
                'pub_key': fake_key,
                'payload': payload,
                'signature': fake_sig,
                'timestamp': timestamp,
            }
        else:
            t0 = time.perf_counter()
            signature = self.sign_message(payload)
            sign_us = (time.perf_counter() - t0) * 1_000_000
            
            print(f"{Colors.OKCYAN}  [NÓ {self.node_id}] Broadcast F={free_energy_delta:.4f} "
                  f"(Sig: {signature[:8].hex()}... | {sign_us:.0f}μs para assinar){Colors.ENDC}")
            
            message = {
                'sender_id': self.node_id,
                'pub_key': self.public_key,
                'payload': payload,
                'signature': signature,
                'timestamp': timestamp,
            }
        
        for node in network_bus:
            if node.node_id != self.node_id:
                node.inbox.put(message)
    
    def run(self):
        while True:
            try:
                msg = self.inbox.get(timeout=3.0)
            except queue.Empty:
                break
            
            # 1. Anti-Replay: verificar timestamp
            age = time.time() - msg['timestamp']
            if age > 2.0 or age < -0.1:
                print(f"{Colors.WARNING}  [NÓ {self.node_id}] 🔴 REJEITADO NÓ {msg['sender_id']}: "
                      f"Replay Attack (pacote com {age:.1f}s de idade){Colors.ENDC}")
                continue
            
            # 2. Verificação criptográfica REAL
            t0 = time.perf_counter()
            is_valid = self.verify_signature(msg['payload'], msg['signature'], msg['pub_key'])
            verify_us = (time.perf_counter() - t0) * 1_000_000
            
            if not is_valid:
                print(f"{Colors.FAIL}  [NÓ {self.node_id}] 🔴 REJEITADO NÓ {msg['sender_id']}: "
                      f"Assinatura Ed25519 INVÁLIDA ({verify_us:.0f}μs para rejeitar){Colors.ENDC}")
                continue
            
            print(f"{Colors.OKGREEN}  [NÓ {self.node_id}] ✅ ACEITO NÓ {msg['sender_id']}: "
                  f"Assinatura válida ({verify_us:.0f}μs para verificar){Colors.ENDC}")


def main():
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"  02. CROM P2P v3 | Criptografia Ed25519 REAL")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
    
    print(f"\n[SYS] Gerando pares de chaves Ed25519 para 3 nós...")
    node1 = P2PNode(node_id=1)
    node2 = P2PNode(node_id=2)
    node3 = P2PNode(node_id=3, is_malicious=True)
    
    print(f"  Nó 1 PubKey: {node1.pub_bytes[:8].hex()}...")
    print(f"  Nó 2 PubKey: {node2.pub_bytes[:8].hex()}...")
    print(f"  Nó 3 PubKey: {node3.pub_bytes[:8].hex()}... (HACKER)")
    
    print(f"\n[SYS] Iniciando rede P2P...\n")
    
    node1.start()
    node2.start()
    node3.start()
    
    # Nó 1 envia delta legítimo
    node1.broadcast(free_energy_delta=0.1523)
    time.sleep(0.3)
    
    # Nó 2 envia delta legítimo
    node2.broadcast(free_energy_delta=0.4201)
    time.sleep(0.3)
    
    # Nó 3 (hacker) tenta injetar estado forjado
    node3.broadcast(free_energy_delta=0.0000)
    
    # Esperar processamento
    node1.join(timeout=5)
    node2.join(timeout=5)
    node3.join(timeout=5)
    
    print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}🔬 ANÁLISE DO CIENTISTA CÉTICO:{Colors.ENDC}")
    print(f"As chaves Ed25519 são curvas elípticas REAIS (não strings hardcoded).")
    print(f"O hacker gerou 64 bytes aleatórios como assinatura — a matemática rejeitou.")
    print(f"Nenhum 'if string == INVALID'. A rejeição vem da álgebra de curva elíptica.")
    print(f"Tempo de verificação: microsegundos (não bloqueia a inferência).")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")


if __name__ == "__main__":
    main()
