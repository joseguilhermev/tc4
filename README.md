# Documentação do Projeto

Este repositório contém um conjunto de scripts em Python, arquivos de configuração Docker e dados para demonstrar um fluxo de trabalho completo de processamento, análise e inferência do modelo LSTM. A seguir, apresentamos uma visão geral rápida para que você possa iniciar rapidamente.

---

## Sumário

1. [Visão Geral](#visão-geral)
2. [Estrutura de Pastas](#estrutura-de-pastas)
3. [Pré-Requisitos](#pré-requisitos)
4. [Instalação](#instalação)
5. [Uso](#uso)
6. [Execução com Docker](#execução-com-docker)
7. [Contribuindo](#contribuindo)
8. [Licença](#licença)

---

## Visão Geral

O objetivo deste projeto é demonstrar como carregar dados, realizar pré-processamento, treinar um modelo de Machine Learning e realizar inferências.  
- **data/**: Contém arquivos CSV e outros dados brutos.  
- **src/**: Scripts em Python para manipular dados, treinar modelos e fazer inferências.  
- **app.py**: API.  
- **Dockerfile** e **compose.yaml**: Arquivos para containerização e orquestração da aplicação.  

---

## Estrutura de Pastas

```
.
├── data
│   └── AAPL
│       └── AAPL.csv            # Exemplo de conjunto de dados
├── src
│   ├── data_processing.py      # Funções de limpeza e preparo de dados
│   ├── inference.py            # Lógica de inferência do modelo
│   ├── model.py                # Definição ou carregamento do modelo
│   └── train.py                # Script principal de treino
├── .gitignore
├── app.py                      # Ponto de entrada para execução do projeto
├── compose.yaml                # Arquivo de configuração do Docker Compose
├── Dockerfile                  # Dockerfile para build da imagem
├── request.py                  # Exemplo de requisições (HTTP, API, etc.)
└── requirements.txt            # Dependências Python
```

---

## Pré-Requisitos

- **Python 3.8+** instalado.
- Gerenciador de pacotes **pip** ou similar (pipenv, poetry, etc.).
- (Opcional) **Docker** e **Docker Compose**, caso deseje executar em contêiner.

---

## Instalação

1. **Clonar o repositório**  
   ```bash
   git clone https://github.com/usuario/projeto.git
   cd projeto
   ```

2. **Instalar dependências**  
   ```bash
   pip install -r requirements.txt
   ```
   Isso instalará todos os pacotes Python necessários, como pandas, numpy, etc.

---

## Uso

### 1. Preparar os dados
Dentro de `src/data_processing.py`, você encontrará funções para limpeza e formatação dos dados. Para executá-las diretamente, use:
```bash
python src/data_processing.py
```
Caso haja parâmetros configuráveis, ajuste-os dentro do arquivo ou via linha de comando (conforme implementação).

### 2. Treinar o modelo
Em `src/train.py`, você encontra o fluxo de treinamento. Para iniciar o treinamento:
```bash
python src/train.py
```
O script lê os dados de `data/`, executa o pré-processamento, treina o modelo e salva o resultado (por exemplo, em um arquivo `.pkl` ou equivalente).

### 3. Fazer inferências
No `src/inference.py`, você encontra a lógica de inferência. Normalmente você chamaria:
```bash
python src/inference.py
```
Certifique-se de que o modelo treinado esteja disponível (geralmente no mesmo diretório ou em local configurado).

### 4. Rodar a aplicação principal
O arquivo `app.py` pode servir como ponto de entrada para executar todo o fluxo (treino, inferência ou inicialização de um serviço web). Por exemplo:
```bash
python app.py
```
A forma exata de uso depende de como o `app.py` foi implementado (CLI, API, etc.).

---

## Execução com Docker

1. **Build da imagem**  
   ```bash
   docker build -t nome-da-imagem .
   ```
2. **Executar o contêiner**  
   ```bash
   docker run -p 8000:8000 nome-da-imagem
   ```
   Isso disponibiliza a aplicação (por exemplo, API ou serviço web) na porta configurada (8000 no exemplo).

3. **Docker Compose**  
   Se preferir usar o `docker-compose`:
   ```bash
   docker-compose up
   ```
   O `compose.yaml` já deve estar configurado com as instruções para buildar a imagem e executar o contêiner.
