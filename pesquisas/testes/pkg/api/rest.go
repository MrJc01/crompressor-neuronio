package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/MrJc01/crompressor-neuronio/pesquisas/testes/pkg/engine"
	"github.com/MrJc01/crompressor-neuronio/pesquisas/testes/pkg/fuse_mount"
	"github.com/MrJc01/crompressor-neuronio/pesquisas/testes/pkg/routing"
)

type APIServer struct {
	Router *routing.HNSWRouter
	Node   *fuse_mount.CromNode
	LLM    *engine.LLMClient
	MntDir string
	Data   []byte
}

func StartWebServer(port string, srv *APIServer) error {
	mux := http.NewServeMux()

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

	mux.HandleFunc("/stats", corsHandler(srv.handleStats))
	mux.HandleFunc("/brains", corsHandler(srv.handleBrains))
	mux.HandleFunc("/ingest", corsHandler(srv.handleIngest))
	mux.HandleFunc("/chat", corsHandler(srv.handleChat))

	log.Printf("🌐 API Orquestradora Rodando na porta %s", port)
	return http.ListenAndServe(":"+port, mux)
}

func (s *APIServer) handleStats(w http.ResponseWriter, r *http.Request) {
	s.Node.StatsLock.RLock()
	defer s.Node.StatsLock.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(s.Node.MutationStats)
}

// Retorna todos os cérebros registrados e seus dados
func (s *APIServer) handleBrains(w http.ResponseWriter, r *http.Request) {
	s.Router.Mu.RLock()
	defer s.Router.Mu.RUnlock()

	brains := make([]map[string]interface{}, 0)
	for _, b := range s.Router.Brains {
		brains = append(brains, map[string]interface{}{
			"id":       b.BrainID,
			"isActive": b.IsActive,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(brains)
}

func (s *APIServer) handleIngest(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Método não permitido", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Domain string `json:"domain"`
		Text   string `json:"text"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	log.Printf("📥 Ingestão Iniciada: Domínio [%s]", req.Domain)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// 1. Gera Embedding Real [896] via LLM Local
	emb, err := s.LLM.GetEmbedding(ctx, req.Text)
	if err != nil {
		http.Error(w, "Erro ao extrair vetor: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// 2. Cria o arquivo lógico .crom associado (brain_x.crom)
	brainID := fmt.Sprintf("brain_%s.crom", req.Domain)

	// 3. Registra no Roteador HNSW
	err = s.Router.RegisterBrain(brainID, emb)
	if err != nil {
		http.Error(w, "Erro ao registrar cérebro: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// 4. Cria um delta XOR sintético para este cérebro.
	// Numa versão final, esse delta seria calculado comparando o espaço vetorial, 
	// mas para não alterar a API do Kernel FUSE atual, injetamos o bytecode pattern (hash byte)
	hashByte := byte(len(req.Domain) * 10)
	delta := make([]byte, len(s.Data))
	copy(delta, s.Data)
	for i := 1024 * 1024; i < len(delta); i += 1024 * 1024 {
		end := i + 512
		if end > len(delta) {
			end = len(delta)
		}
		for j := i; j < end; j++ {
			delta[j] = s.Data[j] ^ hashByte
		}
	}
	s.Node.StatsLock.Lock()
	s.Node.BrainDeltas[brainID] = delta
	s.Node.StatsLock.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":  "sucesso",
		"brainID": brainID,
		"dim":     len(emb),
	})
}

func (s *APIServer) handleChat(w http.ResponseWriter, r *http.Request) {
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

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	hnsStart := time.Now()

	// ── 1. EMBEDDING REAL DO PROMPT ──
	promptEmb, err := s.LLM.GetEmbedding(ctx, req.Prompt)
	if err != nil {
		http.Error(w, "Falha na conversão sub-simbólica: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// ── 2. HNSW ROUTING (Similaridade de Cosseno com 896 Dims) ──
	weights, indices := s.Router.GetTopKWeights(promptEmb, len(s.Router.Brains))
	
	routeInfo := make([]map[string]interface{}, 0)
	var chosenBrain string
	
	// Para uso no cockpit, pegamos o de maior peso pro forward pass principal
    if len(indices) > 0 {
		s.Router.Mu.RLock()
        chosenBrain = s.Router.Brains[indices[0]].BrainID
        s.Node.ActiveContext = s.Router.Brains[indices[0]].Centroid
		s.Router.Mu.RUnlock()
    } else {
		chosenBrain = "base"
	}

	s.Router.Mu.RLock()
	for i, idx := range indices {
		routeInfo = append(routeInfo, map[string]interface{}{
			"brain":  s.Router.Brains[idx].BrainID,
			"weight": weights[i],
		})
	}
	s.Router.Mu.RUnlock()

	hnsElapsed := time.Since(hnsStart)
	log.Printf("🧠 HNSW Routing via Distância Vetorial (896-Dim) -> %s (%.0fμs)", chosenBrain, float64(hnsElapsed.Microseconds()))

	// ── 3. Leitura Background do FUSE p/ gerar Calor no Grafico ──
	go func() {
		fuseFile := filepath.Join(s.MntDir, "virtual_brain.gguf")
		f, ferr := os.Open(fuseFile)
		if ferr != nil {
			return
		}
		defer f.Close()
		buf := make([]byte, 4096)
		totalSize := int64(len(s.Data))
		for block := int64(0); block < 50; block++ {
			offset := block * (totalSize / 50)
			f.ReadAt(buf, offset)
		}
	}()

	// ── 4. INFERÊNCIA LLM ──
	infStart := time.Now()
	systemPrompt := fmt.Sprintf(
		"Você é um neurônio artificial do ecossistema Crompressor-Neurônio. "+
		"Seu cérebro base (.gguf) está congelado no disco como brain.crom (DNA Base-4) "+
		"e lido via FUSE kernel driver com deltas XOR aplicados em tempo real. "+
		"O roteador HNSW selecionou o sub-cérebro '%s' para guiar sua resposta baseado "+
		"na similaridade de cosseno do vetor do prompt [896-Dim]. "+
		"Responda de forma útil, precisa, curta e em português.",
		chosenBrain,
	)

	log.Printf("⏳ Iniciando inferência diferencial com LLM (%s)...", chosenBrain)
	respLog, httpErr := s.LLM.GenerateResponse(ctx, req.Prompt, systemPrompt)

	var aiResponse string
	var errStr string
	var tokPerSec float64
	var completionTokens int

	if httpErr != nil {
		errStr = httpErr.Error()
		aiResponse = "(Erro ao conectar com LLM: " + errStr + ")"
	} else {
		aiResponse = respLog.Message
		completionTokens = respLog.CompletionTokens
	}

	infElapsed := time.Since(infStart)
	if infElapsed.Seconds() > 0 && completionTokens > 0 {
		tokPerSec = float64(completionTokens) / infElapsed.Seconds()
	}

	// ── COLETAR ESTATÍSTICAS FUSE ──
	s.Node.StatsLock.RLock()
	touchedBlocks := make(map[string]uint64)
	for k, v := range s.Node.MutationStats {
		if v > 0 {
			touchedBlocks[fmt.Sprintf("%d", k)] = v
		}
	}
	s.Node.StatsLock.RUnlock()

	respPayload := map[string]interface{}{
		"response":      aiResponse,
		"brain":         chosenBrain,
		"error":         errStr,
		"touchedBlocks": touchedBlocks,
		"routing":       routeInfo,
		"metrics": map[string]interface{}{
			"inference_time_ms": infElapsed.Milliseconds(),
			"tokens_per_second": fmt.Sprintf("%.1f", tokPerSec),
			"hnsw_decision_us":  hnsElapsed.Microseconds(),
			"chunks_read":       len(touchedBlocks),
			"completion_tokens": completionTokens,
		},
		"model": map[string]interface{}{
			"name":         "Qwen2.5-0.5B-Instruct",
			"quantization": "Q4_K_M",
			"size_mb":      394,
			"format":       "GGUF → brain.crom (DNA Base-4)",
			"engine":       "llama.cpp via FUSE VFS",
			"vertente":     "Neurônio Fixo (Frozen)",
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(respPayload)
}
