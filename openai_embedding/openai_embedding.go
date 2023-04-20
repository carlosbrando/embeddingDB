package openai_embedding

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// URL base da API OpenAI
const baseURL = "https://api.openai.com/v1/embeddings"

// EmbeddingRequest é uma estrutura que define os campos necessários para fazer uma solicitação de embedding.
type EmbeddingRequest struct {
	Model string      `json:"model"`          // Nome do modelo que será usado para gerar o embedding
	Input interface{} `json:"input"`          // Texto de entrada para o qual o embedding será gerado
	User  string      `json:"user,omitempty"` // Identificador único opcional do usuário final
}

// EmbeddingResponse é uma estrutura que define os campos presentes na resposta da API de embedding.
type EmbeddingResponse struct {
	Object string `json:"object"`
	Data   []struct {
		Object    string    `json:"object"`
		Embedding []float32 `json:"embedding"` // Vetor de embedding retornado pela API
		Index     int       `json:"index"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

// CreateEmbedding envia uma solicitação para a API OpenAI para gerar um embedding para o texto de entrada fornecido.
// Retorna um ponteiro para uma estrutura EmbeddingResponse e um erro, se houver.
func CreateEmbedding(apiKey string, model string, input interface{}, user string) (*EmbeddingResponse, error) {
	// Cria um cliente HTTP
	client := &http.Client{}

	// Cria uma estrutura EmbeddingRequest com os dados fornecidos
	requestData := &EmbeddingRequest{
		Model: model,
		Input: input,
		User:  user,
	}

	// Converte a estrutura requestData em um objeto JSON
	jsonData, err := json.Marshal(requestData)
	if err != nil {
		return nil, err
	}

	// Cria uma nova solicitação HTTP POST com a URL base, os dados JSON e os cabeçalhos apropriados
	req, err := http.NewRequest("POST", baseURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", apiKey))

	// Envia a solicitação e obtém a resposta
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// Verifica se a resposta tem um status HTTP OK (200)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to create embedding: %s", resp.Status)
	}

	// Lê o corpo da resposta e armazena os dados em uma variável
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	// Cria uma estrutura EmbeddingResponse e preenche-a com os dados da resposta JSON
	var result EmbeddingResponse
	err = json.Unmarshal(body, &result)
	if err != nil {
		return nil, err
	}

	// Retorna um ponteiro para a estrutura result e nenhum erro
	return &result, nil
}
