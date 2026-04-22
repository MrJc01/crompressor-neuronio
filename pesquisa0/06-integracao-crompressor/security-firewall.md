# crompressor-security — Firewall de Realidade

## Papel no Ecossistema 5D

O **crompressor-security** é a **amígdala**: o filtro que impede alucinações de contaminar a realidade base.

## O Problema: Context Bleeding

Em IAs com simulação massiva, é comum o **contágio**: um futuro simulado (não-real) se propaga como vetor inercial legítimo no buffer histórico da identidade factual da IA.

- Uma branch descartada "vaza" para a memória principal
- A IA trata uma simulação como se fosse dado real
- Resultado: alucinação estrutural

## A Solução: Sandboxing Matricial

O crompressor-security isola **buffers heurísticos** que rodam inferências temporárias:

```
┌─────────────────────────────────────────┐
│  REALIDADE BASE (protegida)             │
│  ┌─────────┐  ┌─────────┐              │
│  │ Memória  │  │ Codebook│              │
│  │ Factual  │  │ Core    │              │
│  └─────────┘  └─────────┘              │
│         ▲ FIREWALL (security) ▲         │
├─────────┼─────────────────────┼─────────┤
│  SANDBOX DE SIMULAÇÕES                  │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐          │
│  │ B1 │ │ B2 │ │ B3 │ │ B4 │          │
│  └────┘ └────┘ └────┘ └────┘          │
│  (branches temporárias, isoladas)       │
└─────────────────────────────────────────┘
```

## Mecanismos de Proteção

| Mecanismo | Função |
|:----------|:-------|
| **Delta Ratio Check** | Se a entropia da resposta é incompatível com o domínio → rejeita |
| **Ed25519 Sign/Seal** | Assinatura criptográfica de estados verificados |
| **Merkle Proof** | Prova de integridade de dados ao longo do tempo |
| **Dilithium PQC** | Proteção pós-quântica (futuro) |

## Analogia com a Inferência Ativa

Na fórmula de Friston (F = D_KL - Evidence):
- O security **impõe limites** na D_KL máxima permitida
- Se uma branch diverge demais da realidade → **abort**
- Previne a "esquizofrenia computacional" (branches demais, sem âncora)

## Questões Abertas

- [ ] Como definir o threshold de D_KL para "alucinação vs criatividade"?
- [ ] O sandboxing pode ser implementado como FUSE mount isolado?
- [ ] A segurança pós-quântica (Dilithium) protege contra manipulação de Codebooks?
