package fuse_mount

import (
	"context"
	"sync"
	"syscall"

	gofusefs "github.com/hanwen/go-fuse/v2/fs"
	"github.com/hanwen/go-fuse/v2/fuse"

	"github.com/MrJc01/crompressor-neuronio/pesquisas/testes/pkg/routing"
	"github.com/MrJc01/crompressor-neuronio/pesquisas/testes/pkg/tensor"
)

// CromNode representa o nosso "Cérebro Analítico" operando como se fosse um arquivo puro.
type CromNode struct {
	gofusefs.Inode
	virtualSize   uint64
	frozenData    []byte
	Router        *routing.HNSWRouter
	ActiveContext []float32
	BrainDeltas   map[string][]byte   // Mapa de "Arquivos Delta" baixados na RAM (brainID -> deltaMemory)
	
	// Telemetria (Opção 3)
	MutationStats map[int64]uint64    // Contador de acessos por blocos de 1MB
	StatsLock     sync.RWMutex        // Mutex para evitar race conditions na telemetria
}

// Confirma ao compilador as interfaces do Node no Go-FUSE
var _ = (gofusefs.NodeOpener)((*CromNode)(nil))
var _ = (gofusefs.NodeReader)((*CromNode)(nil))
var _ = (gofusefs.NodeGetattrer)((*CromNode)(nil))

// NewCromNode inicializa um nó FUSE Multi-Brain
func NewCromNode(data []byte, router *routing.HNSWRouter) *CromNode {
	return &CromNode{
		virtualSize:   uint64(len(data)),
		frozenData:    data,
		Router:        router,
		ActiveContext: make([]float32, routing.ContextDim),
		BrainDeltas:   make(map[string][]byte),
		MutationStats: make(map[int64]uint64),
	}
}

// Open é invocado quando um aplicativo de ML tenta abrir o arquivo.
func (n *CromNode) Open(ctx context.Context, openFlags uint32) (gofusefs.FileHandle, uint32, syscall.Errno) {
	return nil, fuse.FOPEN_DIRECT_IO, 0
}

// Read é o interceptor. HNSW O(1) roda aqui.
func (n *CromNode) Read(ctx context.Context, fh gofusefs.FileHandle, dest []byte, off int64) (fuse.ReadResult, syscall.Errno) {
	if off >= int64(n.virtualSize) {
		return fuse.ReadResultData(nil), 0 
	}

	end := int(off) + len(dest)
	if end > int(n.virtualSize) {
		end = int(n.virtualSize)
	}

	// 1. Matéria Base Intocável
	pureChunk := n.frozenData[off:end]

	if n.Router == nil || len(n.Router.Brains) == 0 {
		return fuse.ReadResultData(pureChunk), 0
	}

	// 2. Roteamento Express (HNSW Top-K)
	weights, indices := n.Router.GetTopKWeights(n.ActiveContext, len(n.Router.Brains))

	multiDeltas := make([][]byte, 0, len(weights))
	validWeights := make([]float32, 0, len(weights))

	// 3. Empacota Dicionário para Composição
	for i, idx := range indices {
		if weights[i] <= 0.05 {
			continue // Poda VQ (Sparsity) para peso morto. Ignora Cérebros não harmônicos.
		}

		brainID := n.Router.Brains[idx].BrainID
		
		// Simulando a extração do offset linear das memórias RAM Delta
		if deltaMesh, exists := n.BrainDeltas[brainID]; exists {
			if len(deltaMesh) >= end {
				multiDeltas = append(multiDeltas, deltaMesh[off:end])
				validWeights = append(validWeights, weights[i])
			}
		}
	}

	if len(multiDeltas) == 0 {
		return fuse.ReadResultData(pureChunk), 0
	}

	// 4. Aritmética Plástica Cruzada
	composedChunk, err := tensor.ComposeDeltas(pureChunk, multiDeltas, validWeights)
	if err != nil {
		return fuse.ReadResultData(pureChunk), 0 // Fallback seguro (Robustez)
	}

	// 5. Telemetria Ativa (Delta Heatmap)
	// Chunk LBA (Logical Block Address) em blocos de 1MB
	blockIndex := off / (1024 * 1024)
	n.StatsLock.Lock()
	n.MutationStats[blockIndex]++
	n.StatsLock.Unlock()

	// 6. Retorno com Latência Zero-Copy
	return fuse.ReadResultData(composedChunk), 0
}

// Getattr fornece infos do arquivo para comandos do SO.
func (n *CromNode) Getattr(ctx context.Context, f gofusefs.FileHandle, out *fuse.AttrOut) syscall.Errno {
	out.Mode = 0444 // Leitura Fria
	out.Size = n.virtualSize
	out.Blocks = (n.virtualSize + 511) / 512
	return 0
}
