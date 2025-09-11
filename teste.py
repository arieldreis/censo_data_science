import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

dados = pd.read_csv('dados.csv')
print(dados.head())

# Contagem de valores únicos na coluna AG05
contagem = dados['AG05 – Qual foi a sua menção no último Integradão?'].value_counts().sort_index()
print(contagem)

# Deletar o NaN da coluna target
dados = dados.dropna(subset=['AG05 – Qual foi a sua menção no último Integradão?'])

# Contar os valores NaN na coluna AG05
quantidade_nan = dados['AG05 – Qual foi a sua menção no último Integradão?'].isna().sum()
print(f"A quantidade NaN em AG05: {quantidade_nan}")

# Número de linhas e colunas
print(dados.shape) # 277 linhas e 28 colunas

# coloca os primeiros 4 primeiros caractéres como nomes das variáveis
dados.columns = [col[:4] for col in dados.columns] # irá fazer um loop no arquivo csv
print("Resultado dos 4 primeiros caractéres: ", dados.head(5)) # no head() só irá exibir apenas uma linha.

# Coloca o primeiro caractere de cada resposta como valor de todas as variavéis
dados = dados.astype(str).apply(lambda col: col.str[0])
# lambda é uma função simples para fazer coisas complexas
print(dados)

# verifica os tipos dos dados
print(dados.dtypes) # mostra os tipos de cada coluna

# Converter os tipos de dados das variavéis
for col in dados.columns:
    if col != 'AG05':
        dados[col] = pd.to_numeric(dados[col], errors="coerce").fillna(0).astype(int)

# Converter AG05 para categoria (se for o caso)
dados['AG05'] = dados['AG05'].astype('category')
print(dados.dtypes)

############# Exibindo os atributos numéricos #################

# count: Número de valores não nulos na coluna (exclui NaN)
# mean:  Média aritmética dos valores
# std:   Desvio padrão (medida da dispersão dos dados em relação à média)
# min:   Menor valor da coluna
# 25%:   Primeiro quartil (25% dos dados estão abaixo desse valor)
# 50%:   Mediana (metade dos dados estão abaixo desse valor)
# 75%:   Terceiro quartil (75% dos dados estão abaixo desse valor)
# max:   Maior valor da coluna

dados.select_dtypes('number').describe().transport()
print(dados.select_dtypes)