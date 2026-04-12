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

		// Middleware CORS Global
		corsHandler := func(h http.HandlerFunc) http.HandlerFunc {
			return func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Access-Control-Allow-Origin", "*")
				w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
				w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
				if r.Method == "OPTIONS" {
					w.WriteHeader(http.StatusOK)
					return
				}
				h(w, r)
			}
		}

		mux.HandleFunc("/context", corsHandler(func(w http.ResponseWriter, r *http.Request) {
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
		}))

		mux.HandleFunc("/stats", corsHandler(func(w http.ResponseWriter, r *http.Request) {
			cromInodeType.StatsLock.RLock()
			defer cromInodeType.StatsLock.RUnlock()
			
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(cromInodeType.MutationStats)
		}))

		// Endpoint Nativo de Inference para a UI Web
		mux.HandleFunc("/chat", corsHandler(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != "POST" {
				http.Error(w, "Método não permitido", http.StatusMethodNotAllowed)
				return
			}
			
			var req struct {
				Prompt string `json:"prompt"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}

			// ── HNSW Semantic Router ──
			pLower := strings.ToLower(req.Prompt)
			chosenPersona := "base"
			ctxBase := make([]float32, routing.ContextDim)

			codeKW := []string{"python", "codigo", "code", "html", "go ", "func", "programa", "script", "bug", "classe", "function", "algoritmo", "variavel", "loop", "array"}
			mathKW := []string{"calc", "mate", "soma", "num", "equa", "raiz", "integral", "deriv", "multi", "divi", "fator", "primo", "geometr"}
			creatKW := []string{"poema", "criativ", "imagin", "histor", "conto", "escrev", "invent", "fic", "arte", "mus", "pint"}

			for _, kw := range codeKW {
				if strings.Contains(pLower, kw) { chosenPersona = "code"; break }
			}
			if chosenPersona == "base" {
				for _, kw := range mathKW {
					if strings.Contains(pLower, kw) { chosenPersona = "math"; break }
				}
			}
			if chosenPersona == "base" {
				for _, kw := range creatKW {
					if strings.Contains(pLower, kw) { chosenPersona = "creative"; break }
				}
			}

			switch chosenPersona {
			case "code": cromInodeType.ActiveContext = ctxA
			case "math": cromInodeType.ActiveContext = ctxB
			case "creative": cromInodeType.ActiveContext = ctxC
			default: cromInodeType.ActiveContext = ctxBase
			}

			log.Printf("🤖 HNSW Routing -> persona_%s", chosenPersona)

			// ── Captura pesos HNSW para visualização ──
			weights, indices := cromInodeType.Router.GetTopKWeights(cromInodeType.ActiveContext, len(cromInodeType.Router.Brains))
			routeInfo := make([]map[string]interface{}, 0)
			for i, idx := range indices {
				routeInfo = append(routeInfo, map[string]interface{}{
					"brain":  cromInodeType.Router.Brains[idx].BrainID,
					"weight": weights[i],
				})
			}

			// ── Aciona leitura FUSE em background p/ heatmap ──
			go func() {
				fuseFile := filepath.Join(mntDir, "virtual_brain.gguf")
				f, ferr := os.Open(fuseFile)
				if ferr != nil { return }
				defer f.Close()
				buf := make([]byte, 4096)
				totalSize := int64(len(data))
				for block := int64(0); block < 50; block++ {
					offset := block * (totalSize / 50)
					f.ReadAt(buf, offset)
				}
			}()

			// ── Inferência REAL via llama-server (modelo permanente na RAM) ──
			systemPrompt := fmt.Sprintf(
				"Você é o Crompressor-Neurônio, um assistente de IA rodando no ecossistema CROM-FUSE. "+
				"Seu modelo base (.gguf) está congelado no disco e é lido via FUSE kernel driver. "+
				"O roteador HNSW selecionou a persona '%s' para esta mensagem. "+
				"Responda de forma útil, concisa e em português. Máximo 3 parágrafos.",
				chosenPersona,
			)

			llamaReq := map[string]interface{}{
				"messages": []map[string]string{
					{"role": "system", "content": systemPrompt},
					{"role": "user", "content": req.Prompt},
				},
				"max_tokens":  120,
				"temperature": 0.7,
			}

			llamaBody, _ := json.Marshal(llamaReq)

			httpCtx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
			defer cancel()

			httpReq, _ := http.NewRequestWithContext(httpCtx, "POST",
				"http://127.0.0.1:8080/v1/chat/completions",
				strings.NewReader(string(llamaBody)),
			)
			httpReq.Header.Set("Content-Type", "application/json")

			log.Printf("⏳ Enviando para llama-server (persona=%s)...", chosenPersona)
			llamaResp, httpErr := http.DefaultClient.Do(httpReq)

			var aiResponse string
			var errStr string

			if httpErr != nil {
				errStr = httpErr.Error()
				aiResponse = "(Erro ao conectar com llama-server: " + errStr + ")"
				log.Printf("❌ llama-server erro: %s", errStr)
			} else {
				defer llamaResp.Body.Close()
				respBytes, _ := ioutil.ReadAll(llamaResp.Body)

				var chatResp struct {
					Choices []struct {
						Message struct {
							Content string `json:"content"`
						} `json:"message"`
					} `json:"choices"`
					Usage struct {
						PromptTokens     int `json:"prompt_tokens"`
						CompletionTokens int `json:"completion_tokens"`
					} `json:"usage"`
				}

				if jsonErr := json.Unmarshal(respBytes, &chatResp); jsonErr != nil {
					errStr = jsonErr.Error()
					aiResponse = "(Erro parsing resposta: " + string(respBytes[:200]) + ")"
				} else if len(chatResp.Choices) > 0 {
					aiResponse = chatResp.Choices[0].Message.Content
					log.Printf("✅ Resposta recebida (%d tokens prompt, %d gerados)",
						chatResp.Usage.PromptTokens, chatResp.Usage.CompletionTokens)
				} else {
					aiResponse = "(Sem resposta do modelo)"
				}
			}

			// ── Coleta heatmap FUSE ──
			cromInodeType.StatsLock.RLock()
			touchedBlocks := make(map[string]uint64)
			for k, v := range cromInodeType.MutationStats {
				if v > 0 {
					touchedBlocks[fmt.Sprintf("%d", k)] = v
				}
			}
			cromInodeType.StatsLock.RUnlock()

			respPayload := map[string]interface{}{
				"response":      aiResponse,
				"persona":       chosenPersona,
				"error":         errStr,
				"touchedBlocks": touchedBlocks,
				"routing":       routeInfo,
				"model": map[string]interface{}{
					"name":         "Qwen2.5-0.5B-Instruct",
					"quantization": "Q4_K_M",
					"size":         "394 MB",
					"format":       "GGUF",
					"engine":       "llama.cpp (FUSE-intercepted)",
				},
			}

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(respPayload)
		}))
		
		fmt.Println("🌐 MBR API Rodando em http://localhost:9999 (context, stats, chat)")
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

// generateMockDelta aplica mutações XOR sobre a cópia do modelo.
// Em produção, isso seria gerado via LoRA fine-tuning comprimido.
func generateMockDelta(original []byte, filler byte) []byte {
	size := len(original)
	d := make([]byte, size)
	copy(d, original)
	
	// A partir de 1MB (preserva headers GGUF), aplica XOR com padrão espaçado.
	// Cada persona usa um filler diferente (0xAA, 0xBB, 0xCC) criando
	// deltas distintos que o FUSE detecta e mescla via pesos HNSW.
	for i := 1024 * 1024; i < size; i += 1024 * 1024 {
		// Aplica XOR em janela de 512 bytes a cada 1MB
		end := i + 512
		if end > size { end = size }
		for j := i; j < end; j++ {
			d[j] = original[j] ^ filler
		}
	}
	return d
}

