package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/jotfs/fastcdc-go"

	neuronio "github.com/MrJc01/crompressor-neuronio/pesquisas/testes/pkg"
)

func main() {
	fmt.Println("╔═══════════════════════════════════════════════════════╗")
	fmt.Println("║   🧠 GGUF PARSER -> CROM (Semantic FastCDC)          ║")
	fmt.Println("║   Engrenagem de fatiamento geométrico e redução      ║")
	fmt.Println("╚═══════════════════════════════════════════════════════╝")

	modelsDir := filepath.Join("..", "modelos")
	if _, err := os.Stat(modelsDir); os.IsNotExist(err) {
		modelsDir = filepath.Join("pesquisas", "modelos")
	}

	ggufFile := filepath.Join(modelsDir, "qwen2.5-0.5b-q4_k_m.gguf")
	file, err := os.Open(ggufFile)
	if err != nil {
		log.Fatalf("Erro ao abrir GGUF: %v\n", err)
	}
	defer file.Close()

	stat, _ := file.Stat()
	fmt.Printf("📦 Lendo Modelo: %s (%.2f MB)\n", filepath.Base(ggufFile), float64(stat.Size())/1024/1024)

	brain := &neuronio.BrainCrom{
		Header: neuronio.CromHeader{
			Magic:        "CROM",
			Version:      2,
			Domain:       "qwen2.5-0.5b-q4",
			OriginalSize: stat.Size(),
		},
	}

	start := time.Now()
	dedupMap := make(map[string]int)
	var totalChunks, uniqueChunks int

	// Opções do FastCDC - Foco no Semantic Chunking nativo de LLMs
	opts := fastcdc.Options{
		MinSize:     512,     // Evitar overhead de ponteiros para dados inúteis minúsculos
		AverageSize: 4096,    // Alvo do Linux VFS Page Size (Bom para FUSE)
		MaxSize:     16384,   // Cap máximo de 16KB 
	}
	
	chunker, _ := fastcdc.NewChunker(file, opts)

	for {
		chunkData, err := chunker.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatalf("Falha FastCDC: %v", err)
		}

		n := len(chunkData.Data)
		totalChunks++

		hashBytes := sha256.Sum256(chunkData.Data)
		hashStr := string(hashBytes[:])

		var chunk neuronio.Chunk

		if refIdx, exists := dedupMap[hashStr]; exists {
			// Duplicate chunk
			chunk = neuronio.Chunk{
				ID:       totalChunks - 1,
				Hash:     fmt.Sprintf("%x", hashBytes),
				IsDelta:  true,
				DeltaRef: refIdx,
				Size:     n,
				// Data remains nil unless we want memory FUSE simulation, which we do here
				Data:     append([]byte(nil), chunkData.Data...), 
			}
		} else {
			// Unique chunk
			uniqueChunks++
			idx := len(brain.Codebook)
			dedupMap[hashStr] = idx

			entropy := neuronio.ShannonEntropy(chunkData.Data)

			// DNA Compression: strings are costly. Emulação física de DNA bits = len(chunkData) / 4 bytes
			// Se o original gasta `n` bytes, o DNA bit-packed gasta `n/4` bytes se houvesse conversão total (base teórica)
			
			var quickHash uint64
			if len(chunkData.Data) >= 8 {
				for i := 0; i < 8; i++ {
					quickHash = (quickHash << 8) | uint64(chunkData.Data[i])
				}
			}

			brain.Codebook = append(brain.Codebook, neuronio.CodeEntry{
				Hash:     quickHash,
				DNA:      "OMITTED-FOR-MEMORY", // No mundo físico, os bits ficam no Data
				RefCount: 1,
			})

			chunk = neuronio.Chunk{
				ID:      totalChunks - 1,
				Hash:    fmt.Sprintf("%x", hashBytes),
				Size:    n,
				Entropy: entropy,
				IsDelta: false,
				Data:    append([]byte(nil), chunkData.Data...),
			}
		}

		brain.Chunks = append(brain.Chunks, chunk)
	}

	brain.Header.ChunkCount = len(brain.Chunks)
	brain.Header.CodebookSize = len(brain.Codebook)
	brain.Header.Frozen = true

	elapsed := time.Since(start)

	dedupRate := float64(totalChunks-uniqueChunks) / float64(totalChunks) * 100
	rawSize := stat.Size()
	
	// Cálculo Físico Correto de Overhead
	// O Codebook real salva os dados reais `n` bytes, não `n*4`. E nós não salvamos os bytes repetidos.
	var pureCodebookSizeBytes int
	for _, c := range brain.Chunks {
		if !c.IsDelta {
			pureCodebookSizeBytes += c.Size
		}
	}
	
	pointersOverhead := int64(len(brain.Chunks) * 36) // 32 hash + 4 int (referências FUSE)
	cromSize := int64(pureCodebookSizeBytes) + pointersOverhead
	
	brain.Header.CompressedSize = cromSize
	ratio := float64(rawSize) / float64(cromSize)

	fmt.Printf("✅ FastCDC Parsing completo em %v\n", elapsed)
	fmt.Printf("📊 Total Chunks (Dinâmicos): %d\n", totalChunks)
	fmt.Printf("🧬 Chunks Únicos: %d\n", uniqueChunks)
	fmt.Printf("♻️  Taxa de Deduplicação GGUF Real: %.2f%%\n", dedupRate)
	fmt.Printf("📉 Compressão Projetada CROM: %.2fx (de %.2f MB para %.2f MB)\n", 
		ratio, float64(rawSize)/1024/1024, float64(cromSize)/1024/1024)

	// Salvando Metadata
	metaInfo := map[string]interface{}{
		"model": "qwen2.5-0.5b-q4",
		"cdc_strategy": "fastcdc",
		"dedup_rate": dedupRate,
		"original_mb": float64(rawSize)/1024/1024,
		"projected_mb": float64(cromSize)/1024/1024,
		"compression_ratio": ratio,
		"chunks": totalChunks,
	}
	cromOut := filepath.Join(modelsDir, "qwen_parsed_fastcdc.metadata")
	outBytes, _ := json.MarshalIndent(metaInfo, "", "  ")
	os.WriteFile(cromOut, outBytes, 0644)
}
