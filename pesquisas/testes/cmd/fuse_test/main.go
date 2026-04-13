package main

import (
	"context"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/MrJc01/crompressor-neuronio/pesquisas/testes/pkg/api"
	"github.com/MrJc01/crompressor-neuronio/pesquisas/testes/pkg/engine"
	"github.com/MrJc01/crompressor-neuronio/pesquisas/testes/pkg/fuse_mount"
	"github.com/MrJc01/crompressor-neuronio/pesquisas/testes/pkg/routing"

	"github.com/hanwen/go-fuse/v2/fs"
	"github.com/hanwen/go-fuse/v2/fuse"
)

func main() {
	fmt.Println("╔═══════════════════════════════════════════════════════╗")
	fmt.Println("║   🧩 CROM-FUSE VFS & LLM EMBEDDING ENGINE             ║")
	fmt.Println("║   Montando cérebro virtual e injetando conhecimentos  ║")
	fmt.Println("╚═══════════════════════════════════════════════════════╝")

	// 1. Determinação de Caminhos
	modelsDir := filepath.Join("..", "modelos")
	if _, err := os.Stat(modelsDir); os.IsNotExist(err) {
		modelsDir = filepath.Join("pesquisas", "modelos")
	}

	modelFile := filepath.Join(modelsDir, "qwen2.5-0.5b-instruct-q4_k_m.gguf")
	mntDir := filepath.Join(filepath.Dir(modelsDir), "fuse_mnt")

	if _, err := os.Stat(modelFile); os.IsNotExist(err) {
		log.Fatalf("❌ Modelo base não encontrado em %s", modelFile)
	}

	// 2. Carregamento em RAM (PoC)
	fmt.Printf("⏳ Carregando malha base para a RAM (%.2f MB)...\n", 260.37)
	start := time.Now()
	data, err := ioutil.ReadFile(modelFile)
	if err != nil {
		log.Fatalf("Erro ao ler o modelo físico: %v", err)
	}
	fmt.Printf("✅ Malha na RAM em %v\n", time.Since(start))

	// 3. Setup do LLM
	llmClient := engine.NewLLMClient("http://127.0.0.1:8080")

	// 4. Setup Roteamento e Ingestão de Cérebro Base
	fmt.Println("⏳ Injetando Cérebros Iniciaiss HNSW via LLM Embeddings (896-Dim)...")
	router := routing.NewRouter()

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	// Cria tensores reais a partir do LLM
	fmt.Print("  - Injetando brain_codigo.crom... ")
	embCode, _ := llmClient.GetEmbedding(ctx, "programação python golang javascript compiladores")
	if embCode == nil {
		fmt.Println("⚠️ Usando fallback (LLM não respondeu)")
		embCode = make([]float32, routing.ContextDim)
		embCode[0] = 1.0
	} else {
		fmt.Println("✅ 896 Dims")
	}

	fmt.Print("  - Injetando brain_matematica.crom... ")
	embMath, _ := llmClient.GetEmbedding(ctx, "calculo geometria trigonometria matematica estatistica algebra")
	if embMath == nil {
		embMath = make([]float32, routing.ContextDim)
	} else {
		fmt.Println("✅ 896 Dims")
	}

	fmt.Print("  - Injetando brain_literario.crom... ")
	embLit, _ := llmClient.GetEmbedding(ctx, "poemas arte criatividade literatura filosofia contos")
	if embLit == nil {
		embLit = make([]float32, routing.ContextDim)
	} else {
		fmt.Println("✅ 896 Dims")
	}

	router.RegisterBrain("brain_codigo.crom", embCode)
	router.RegisterBrain("brain_matematica.crom", embMath)
	router.RegisterBrain("brain_literario.crom", embLit)

	cromNode := fuse_mount.NewCromNode(data, router)

	// Inject syntethic Byte pattern for these 3 for FUSE rendering (since this is just graphic in PoC)
	cromNode.BrainDeltas["brain_codigo.crom"] = generateDeltaPattern(data, 0xAA)
	cromNode.BrainDeltas["brain_matematica.crom"] = generateDeltaPattern(data, 0xBB)
	cromNode.BrainDeltas["brain_literario.crom"] = generateDeltaPattern(data, 0xCC)

	// 5. Configuração FUSE
	os.MkdirAll(mntDir, 0755)
	opts := &fs.Options{
		MountOptions: fuse.MountOptions{
			Name:    "crom-mbr",
			Options: []string{"ro"},
		},
	}

	root := &fs.Inode{}
	server, err := fs.Mount(mntDir, root, opts)
	if err != nil {
		log.Fatalf("❌ Erro ao anexar FUSE: %v", err)
	}
	root.AddChild("virtual_brain.gguf", root.NewPersistentInode(
		context.Background(), cromNode, fs.StableAttr{Mode: fuse.S_IFREG | 0444},
	), false)
	fmt.Printf("🚀 FUSE ONLINE: %s/virtual_brain.gguf\n\n", mntDir)

	// 6. Iniciar Servidor Web
	apiServer := &api.APIServer{
		Router: router,
		Node:   cromNode,
		LLM:    llmClient,
		MntDir: mntDir,
		Data:   data,
	}

	go func() {
		if err := api.StartWebServer("9999", apiServer); err != nil {
			log.Fatalf("❌ Server falhou: %v", err)
		}
	}()

	fmt.Println("Pressione CTRL+C para desligar o motor de interseção e ejetar.")
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	<-c

	fmt.Println("\n🛑 Encerrando Módulo FUSE... Ejetando ponteiros.")
	server.Unmount()
}

// Em produção, isso seria gerado via treinamento XOR real no kernel.
func generateDeltaPattern(original []byte, filler byte) []byte {
	size := len(original)
	d := make([]byte, size)
	copy(d, original)
	for i := 1024 * 1024; i < size; i += 1024 * 1024 {
		end := i + 512
		if end > size {
			end = size
		}
		for j := i; j < end; j++ {
			d[j] = original[j] ^ filler
		}
	}
	return d
}
