# Embedding Database and Word Similarity Search
Este repositório contém um programa em Go que demonstra o uso de um banco de dados de embeddings para realizar busca de similaridade de palavras usando o algoritmo de similaridade do cosseno. O programa utiliza os pacotes "gonum.org/v1/gonum" e "gonum.org/v1/gonum/floats" para realizar operações matemáticas em matrizes e vetores.

## Uso
O programa define uma estrutura de dados EmbeddingDB que contém um mapa de embeddings de palavras e seus vetores correspondentes. A função NewEmbeddingDB é usada para criar uma nova estrutura EmbeddingDB, enquanto os métodos AddEmbedding e GetEmbedding são usados para adicionar novos embeddings de palavras ao banco de dados e recuperar embeddings existentes, respectivamente.

O programa define uma função CosineSimilarity que calcula a similaridade do cosseno entre dois vetores. A função recebe duas matrizes como entrada, converte-as em vetores de fatia, calcula o produto escalar e a norma L2 de cada vetor e, em seguida, retorna a similaridade do cosseno como o produto escalar dividido pelo produto das normas L2.

O programa também define um método FindClosest que encontra as n palavras mais próximas de uma palavra dada no banco de dados de embeddings usando a função de similaridade do cosseno. O método itera sobre todas as palavras no banco de dados, calcula a similaridade do cosseno em relação à palavra de entrada e armazena as n palavras mais próximas em uma lista. A lista é então ordenada por similaridade e as n palavras mais próximas são retornadas.

O programa principal cria uma nova estrutura EmbeddingDB, adiciona vários embeddings de palavras ao banco de dados e usa o método FindClosest para encontrar as três palavras mais próximas da palavra "apple" no banco de dados. Os resultados são então impressos no console.

Para executar o programa, certifique-se de ter o Go instalado e execute o seguinte comando na pasta do projeto:

```go
go run main.go
```

## Contribuindo
Contribuições são bem-vindas! Sinta-se à vontade para abrir uma issue ou enviar um pull request.

## Licença
Este programa é licenciado sob a licença MIT. Consulte o arquivo LICENSE para obter mais informações.
