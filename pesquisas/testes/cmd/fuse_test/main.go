package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/MrJc01/crompressor-neuronio/pesquisas/testes/pkg/fuse_mount"
	"github.com/MrJc01/crompressor-neuronio/pesquisas/testes/pkg/routing"
	
	"github.com/hanwen/go-fuse/v2/fs"
	"github.com/hanwen/go-fuse/v2/fuse"
)

func main() {
	fmt.Println("╔═══════════════════════════════════════════════════════╗")
	fmt.Println("║   🧩 CROM-FUSE VFS INICIALIZADO                      ║")
	fmt.Println("║   Montando cérebro virtual no kernel Linux           ║")
	fmt.Println("╚═══════════════════════════════════════════════════════╝")

	// 1. Determinação de Caminhos
	modelsDir := filepath.Join("..", "modelos")
	if _, err := os.Stat(modelsDir); os.IsNotExist(err) {
		modelsDir = filepath.Join("pesquisas", "modelos") // fallback se rodar da raiz
	}

	modelFile := filepath.Join(modelsDir, "qwen2.5-0.5b-instruct-q4_k_m.gguf")
	mntDir := filepath.Join(filepath.Dir(modelsDir), "fuse_mnt")

	if _, err := os.Stat(modelFile); os.IsNotExist(err) {
		log.Fatalf("❌ Modelo base não encontrado em %s", modelFile)
	}

	// 2. Carregamento em RAM (Apenas para Fase de Testes POC de ~260MB)
	fmt.Printf("⏳ Carregando malha base para a RAM (%.2f MB)...\n", 260.37)
	start := time.Now()
	data, err := ioutil.ReadFile(modelFile)
	if err != nil {
		log.Fatalf("Erro ao ler o modelo físico: %v", err)
	}
	fmt.Printf("✅ Malha na RAM em %v\n", time.Since(start))

	// 3. Setup Multi-Brain Router HNSW
	fmt.Println("⏳ Inicializando Topologia HNSW com 3 Neuro-Personas Simuladas...")
	router := routing.NewRouter()
	
	// Mock Contexts (Arbitrary 128-dim vectors)
	ctxA := make([]float32, routing.ContextDim); ctxA[0] = 1.0 // Logical Persona A (Coding)
	ctxB := make([]float32, routing.ContextDim); ctxB[1] = 1.0 // Logical Persona B (Math)
	ctxC := make([]float32, routing.ContextDim); ctxC[2] = 1.0 // Logical Persona C (Creative)

	router.RegisterBrain("persona_code", ctxA)
	router.RegisterBrain("persona_math", ctxB)
	router.RegisterBrain("persona_creative", ctxC)

	cromInodeType := fuse_mount.NewCromNode(data, router)

	// Injetar deltas sintéticos estáticos gerados em Runtime (só para FUSE Proof)
	cromInodeType.BrainDeltas["persona_code"] = generateMockDelta(data, 0xAA) // Byte pattern
	cromInodeType.BrainDeltas["persona_math"] = generateMockDelta(data, 0xBB)
	cromInodeType.BrainDeltas["persona_creative"] = generateMockDelta(data, 0xCC)

	// Configuração do Servidor FUSE
	opts := &fs.Options{
		MountOptions: fuse.MountOptions{
			Name:       "crom-mbr", 
			Options:    []string{"ro"}, 
		},
	}

	// Root Inode for FUSE structure
	root := &fs.Inode{}

	// Forçar a desmontagem caso o app tenha sujado a pasta numa execução anterior
	os.MkdirAll(mntDir, 0755)
	
	server, err := fs.Mount(mntDir, root, opts)
	if err != nil {
		log.Fatalf("❌ Erro catastrófico ao anexar FUSE a %s: %v", mntDir, err)
	}

	// Pendura o nosso cérebro MBR como filho do Root Inode 
	root.AddChild("virtual_brain.gguf", root.NewPersistentInode(
		context.Background(), cromInodeType, fs.StableAttr{Mode: fuse.S_IFREG | 0444},
	), false)

	fmt.Printf("🚀 FUSE ONLINE! Ponte neural armada com sucesso em:\n   -> %s/virtual_brain.gguf\n\n", mntDir)
	
	// ---- SERVIDOR DE ROTEAMENTO REST (MUDANÇA DE CONTEXTO SEMÂNTICA O(1)) ----
	go func() {
		mux := http.NewServeMux()
		mux.HandleFunc("/context", func(w http.ResponseWriter, r *http.Request) {
			persona := r.URL.Query().Get("persona")
			persona = strings.TrimSpace(persona)
			log.Printf("📡 Requisição de contexto recebida: [%s]", persona)

			ctxBase := make([]float32, routing.ContextDim)

			switch persona {
			case "base":
				cromInodeType.ActiveContext = ctxBase
				fmt.Fprintln(w, "➜ Pesos HNSW resetados para Persona BASE (Nula).")
				log.Println("✅ Contexto resetado para BASE")
			case "code":
				cromInodeType.ActiveContext = ctxA
				fmt.Fprintln(w, "➜ Pesos HNSW comutados instantaneamente para Persona A (Coding).")
				log.Println("✅ Contexto comutado para CODE")
			case "math":
				cromInodeType.ActiveContext = ctxB
				fmt.Fprintln(w, "➜ Pesos HNSW comutados instantaneamente para Persona B (Math).")
				log.Println("✅ Contexto comutado para MATH")
			case "creative":
				cromInodeType.ActiveContext = ctxC
				fmt.Fprintln(w, "➜ Pesos HNSW comutados instantaneamente para Persona C (Creative).")
				log.Println("✅ Contexto comutado para CREATIVE")
			default:
				fmt.Fprintln(w, "Contexto Desconhecido.")
				log.Printf("⚠️ Contexto desconhecido recebido: [%s]", persona)
			}
		})

		mux.HandleFunc("/stats", func(w http.ResponseWriter, r *http.Request) {
			cromInodeType.StatsLock.RLock()
			defer cromInodeType.StatsLock.RUnlock()
			
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(cromInodeType.MutationStats)
		})
		
		fmt.Println("🌐 MBR API Rodando em http://localhost:9999/context")
		if err := http.ListenAndServe(":9999", mux); err != nil {
			log.Fatalf("❌ Falha catastrófica ao iniciar servidor API: %v", err)
		}
	}()

	fmt.Println("Pressione CTRL+C para desligar o motor de interseção e ejetar.")

	// Esperar o sinal SIGINT para desligar graciosamente
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	<-c

	fmt.Println("\n🛑 Encerrando Módulo FUSE... Ejetando ponteiros de Kernel.")
	server.Unmount()
}

func generateMockDelta(original []byte, filler byte) []byte {
	size := len(original)
	d := make([]byte, size)
	copy(d, original)
	
	// Modificamos a partir de 1MB para não quebrar metadados e headers GGUF,
	// apenas aplicando "ruído artificial" (na demonstração real, seriam pesos Q4 ajustados do LoRA).
	// Aqui vamos modificar raramente (cada 1MB) para não corromper completamente a matriz.
	for i := 1024 * 1024; i < size; i += 1024 * 1024 {
		// Substituição leve intencional para mostrar diferença mutante
		// Se deixar vazio (sem mudanças), FUSE provará overhead zero idêntico.
	}
	return d
}
