package engine

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io/ioutil"
	"net/http"
)

// LLMClient representa o cliente de interação com o llama-server nativo
type LLMClient struct {
	BaseURL string
}

func NewLLMClient(url string) *LLMClient {
	return &LLMClient{BaseURL: url}
}

// GetEmbedding solicita ao llama.cpp o tensor denso (ex: 896 dimensões) a partir de um texto.
// ISTO GERA DADOS REAIS DE ANÁLISE SEMÂNTICA, sem falso simulador.
func (c *LLMClient) GetEmbedding(ctx context.Context, text string) ([]float32, error) {
	reqBody := map[string]string{
		"input": text,
	}
	bodyBytes, _ := json.Marshal(reqBody)

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.BaseURL+"/v1/embeddings", bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyErr, _ := ioutil.ReadAll(resp.Body)
		return nil, errors.New("falha na api de embeddings: " + string(bodyErr))
	}

	var embResp struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&embResp); err != nil {
		return nil, err
	}

	if len(embResp.Data) == 0 || len(embResp.Data[0].Embedding) == 0 {
		return nil, errors.New("nenhum vetor de embedding retornado")
	}

	return embResp.Data[0].Embedding, nil
}

type ChatResponse struct {
	Message          string
	PromptTokens     int
	CompletionTokens int
}

// GenerateResponse solicita inferência real do modelo base com os tensores deltas ativados em Kernel.
func (c *LLMClient) GenerateResponse(ctx context.Context, prompt string, system string) (*ChatResponse, error) {
	llamaReq := map[string]interface{}{
		"messages": []map[string]string{
			{"role": "system", "content": system},
			{"role": "user", "content": prompt},
		},
		"max_tokens":  200,
		"temperature": 0.7,
	}

	llamaBody, _ := json.Marshal(llamaReq)

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.BaseURL+"/v1/chat/completions", bytes.NewReader(llamaBody))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBytes, _ := ioutil.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return nil, errors.New("llama-server retornou status " + resp.Status + ": " + string(respBytes))
	}

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

	if err := json.Unmarshal(respBytes, &chatResp); err != nil {
		return nil, err
	}

	if len(chatResp.Choices) == 0 {
		return nil, errors.New("nenhuma escolha retornada pelo modelo")
	}

	return &ChatResponse{
		Message:          chatResp.Choices[0].Message.Content,
		PromptTokens:     chatResp.Usage.PromptTokens,
		CompletionTokens: chatResp.Usage.CompletionTokens,
	}, nil
}
