# 🛍️ Black Friday Analysis
Projeto de Machine Learning para análise de comportamento de consumo e segmentação de clientes com base em dados de compras da Black Friday.

## 📊 Objetivo
Responder à pergunta de negócio:
Quais segmentos de clientes gastam mais?
Além disso, o projeto constrói um modelo preditivo que classifica clientes em três categorias de gasto:

Baixo
Médio
Alto


## 📁 Dataset
Este projeto utiliza um dataset de vendas de varejo da Black Friday contendo aproximadamente 50 mil registros.

Nome esperado do arquivo:
dataset_retail_black_friday_sales_50k.xlsx
Como usar seu próprio dataset
Você pode substituir por outro dataset desde que possua colunas semelhantes, como:

segmento_cliente
valor_compra
data_compra

Ou adaptar o código conforme necessário.

## ⚙️ Tecnologias utilizadas

Python 3.x
Pandas
NumPy
Scikit-learn


## 🔍 Pipeline do projeto
### 1. Leitura e análise inicial

Visualização das primeiras linhas
Verificação de tipos de dados
Dimensão do dataset


### 2. Análise de negócio
Agrupamento por segmento de cliente para identificar:

Quantidade de compras
Gasto médio
Gasto mediano
Gasto total


### 3. Engenharia de features
iação da variável alvo faixa_gasto com três categorias: Baixo, Médio e Alto, utilizando divisão por quantis.
Também são criadas variáveis derivadas da data:

Ano
Mês
Dia
Dia da semana


### 4. Pré-processamento

Tratamento de valores ausentes
Codificação de variáveis categóricas (One-Hot Encoding)
Normalização (para regressão logística)


### 5. Modelos aplicados

Árvore de Decisão
Random Forest
Regressão Logística


### 6. Avaliação dos modelos
Métricas utilizadas:

Acurácia
Precisão
Recall
F1-score
Matriz de confusão


## 🏆 Saída do modelo
O projeto gera uma tabela final com:

Desempenho de cada algoritmo
Tempo de treinamento
Comparação geral


## ▶️ Como executar

Clone o repositório

git clone [https://github.com/GabrielNonnemacher/black-friday-analysis](https://github.com/GabrielNonnemacher/black-friday-analysis)

Instale as dependências

pip install pandas numpy scikit-learn openpyxl

Adicione o dataset na raiz do projeto com o nome:

dataset_retail_black_friday_sales_50k.xlsx

Execute o projeto

python main.py

## 📌 Observações técnicas

Colunas que causam data leakage foram removidas (ex: valor da compra, IDs)
A variável alvo é derivada diretamente do valor de compra
O uso de stratify no treino garante equilíbrio entre classes


## 🚀 Possíveis melhorias

Validação cruzada (cross-validation)
Ajuste de hiperparâmetros (GridSearchCV)
Análise de importância das features
Deploy com API (FastAPI)
Dashboard com Streamlit

 
## 👨‍💻 Autor
  - Gabriel José Nonnemacher
  - Lucas Teixeira

## 📄 Licença
Projeto com finalidade educacional
