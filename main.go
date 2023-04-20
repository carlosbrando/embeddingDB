package main

import (
	"bufio"
	"fmt" // Para saída formatada
	"os"
	"strings"

	"github.com/carlosbrando/embedding-db/openai_embedding"

	// Para gerar números aleatórios
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat" // Para realizar operações matemáticas
)

// TODO: Trocar por ENV vars depois
const model = "text-embedding-ada-002"

// Estrutura do banco de dados de embeddings
type EmbeddingDB struct {
	embeddings map[string]*mat.Dense // Mapa para armazenar os embeddings das palavras
	dim        int                   // Dimensão dos embeddings
}

// Função para criar um novo banco de dados de embeddings
func NewEmbeddingDB(dim int) *EmbeddingDB {
	return &EmbeddingDB{
		embeddings: make(map[string]*mat.Dense), // Inicializar o mapa de embeddings
		dim:        dim,                         // Definir a dimensão dos embeddings
	}
}

// Função para adicionar um novo embedding ao banco de dados
func (db *EmbeddingDB) AddEmbedding(word string, apiKey string) {
	// Solicitar um embedding da API OpenAI usando nosso pacote openai_embedding
	embeddingResponse, err := openai_embedding.CreateEmbedding(apiKey, model, word, "")
	if err != nil {
		fmt.Printf("Error creating embedding for word '%s': %v\n", word, err)
		return
	}

	// Obter o vetor de embedding gerado pela API OpenAI
	openaiEmbedding := embeddingResponse.Data[0].Embedding

	// Converter []float32 para []float64
	embedding := make([]float64, len(openaiEmbedding))
	for i, value := range openaiEmbedding {
		embedding[i] = float64(value)
	}

	// Adicionar o embedding gerado ao banco de dados
	db.embeddings[word] = mat.NewDense(1, db.dim, embedding)

}

// Função para obter o embedding de uma palavra do banco de dados
func (db *EmbeddingDB) GetEmbedding(word string) *mat.Dense {
	return db.embeddings[word] // Retornar o embedding da palavra solicitada
}

// CosineSimilarity calcula a similaridade do cosseno entre dois vetores de entrada,
// que são representados como matrizes de uma linha (vetores linha).
func CosineSimilarity(vec1, vec2 *mat.Dense) float64 {
	// Obtenha as dimensões das matrizes de entrada.
	row1, _ := vec1.Dims()
	row2, _ := vec2.Dims()

	// Verifique se ambas as entradas são vetores linha.
	if row1 != 1 || row2 != 1 {
		panic("CosineSimilarity: both inputs must be row vectors")
	}

	// Converta as matrizes de entrada em vetores de fatia.
	v1 := mat.Row(nil, 0, vec1)
	v2 := mat.Row(nil, 0, vec2)

	// Calcule o produto escalar dos vetores.
	dotProduct := floats.Dot(v1, v2)
	// Calcule a norma L2 (magnitude) de cada vetor.
	norm1 := floats.Norm(v1, 2)
	norm2 := floats.Norm(v2, 2)

	// Calcule e retorne a similaridade do cosseno, que é o produto escalar
	// dividido pelo produto das magnitudes dos vetores.
	return dotProduct / (norm1 * norm2)
}

// FindClosest encontra as n palavras mais próximas à palavra de entrada no banco de dados de embeddings.
// A similaridade entre as palavras é calculada usando a similaridade do cosseno.
func (db *EmbeddingDB) FindClosest(word string, n int) []string {
	// Obter o embedding da palavra de entrada
	inputEmbedding := db.GetEmbedding(word)

	// Inicializar uma lista de palavras próximas e seus scores de similaridade
	type wordSimilarity struct {
		word       string
		similarity float64
	}
	closestWords := make([]wordSimilarity, 0, n)

	// Iterar sobre todas as palavras e seus embeddings no banco de dados
	for w, embedding := range db.embeddings {
		// Calcular a similaridade do cosseno entre o embedding da palavra de entrada e o embedding atual
		cosineSimilarity := CosineSimilarity(inputEmbedding, embedding)

		// Inserir a palavra e sua similaridade na lista de palavras próximas, se necessário
		if len(closestWords) < n || cosineSimilarity > closestWords[0].similarity {
			// Criar uma nova entrada wordSimilarity para a palavra e sua similaridade
			newEntry := wordSimilarity{word: w, similarity: cosineSimilarity}

			// Encontrar a posição adequada para inserir a nova entrada na lista
			pos := 0
			for pos < len(closestWords) && cosineSimilarity > closestWords[pos].similarity {
				pos++
			}

			// Inserir a nova entrada na lista na posição correta
			closestWords = append(closestWords[:pos], append([]wordSimilarity{newEntry}, closestWords[pos:]...)...)

			// Remover a primeira entrada da lista (a palavra com menor similaridade) se a lista tiver mais de n elementos
			if len(closestWords) > n {
				closestWords = closestWords[1:]
			}
		}
	}

	// Extrair e retornar apenas os nomes das palavras próximas (ignorando os scores de similaridade)
	result := make([]string, len(closestWords))
	for i, entry := range closestWords {
		result[i] = entry.word
	}
	return result
}

func main() {
	apiKey := os.Getenv("API_KEY")
	if apiKey == "" {
		fmt.Println("A variável de ambiente 'EMBEDDING_DB_API_KEY' não está definida.")
		os.Exit(1)
	}

	db := NewEmbeddingDB(1536) // openai tem 1536 dimensões

	words := []string{"maça", "banana", "laranja", "uva", "morango", "abacaxi"}
	for _, word := range words {
		db.AddEmbedding(word, apiKey)
	}

	reader := bufio.NewReader(os.Stdin) // Leitor para ler a entrada do terminal

	for {
		fmt.Print("Digite uma palavra (ou 'sair' para encerrar): ")
		input, _ := reader.ReadString('\n')               // Ler a entrada do usuário
		input = strings.TrimSpace(strings.ToLower(input)) // Remover espaços e converter para minúsculas

		if input == "sair" {
			break
		}

		// Adicionar a palavra de entrada ao banco de dados se ainda não estiver presente
		if _, ok := db.embeddings[input]; !ok {
			db.AddEmbedding(input, apiKey)
		}

		// Encontrar e exibir as palavras mais próximas do dicionário
		closestWords := db.FindClosest(input, 5)
		fmt.Printf("Palavras mais próximas de '%s': %v\n\n", input, closestWords)
	}
}
