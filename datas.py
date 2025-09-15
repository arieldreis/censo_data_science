import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


dadosFrame= pd.read_csv('dados.csv')
print(dadosFrame.head())

# Contagem de valores únicos na coluna AG05
contagem = dadosFrame['AG05 – Qual foi a sua menção no último Integradão?'].value_counts().sort_index()
print(contagem)

# Deletar o NaN da coluna target
dados = dadosFrame.dropna(subset=['AG05 – Qual foi a sua menção no último Integradão?'])

# Contar os valores NaN na coluna AG05
quantidade_nan = dadosFrame['AG05 – Qual foi a sua menção no último Integradão?'].isna().sum()
print(f"A quantidade NaN em AG05: {quantidade_nan}")

# Número de linhas e colunas
print(dadosFrame.shape) # 277 linhas e 28 colunas

# coloca os primeiros 4 primeiros caractéres como nomes das variáveis
dadosFrame.columns = [col[:4] for col in dadosFrame.columns] # irá fazer um loop no arquivo csv
print("Resultado dos 4 primeiros caractéres: ", dados.head(5)) # no head() só irá exibir apenas uma linha.

# Coloca o primeiro caractere de cada resposta como valor de todas as variavéis
dadosFrame = dadosFrame.astype(str).apply(lambda col: col.str[0])
# lambda é uma função simples para fazer coisas complexas
print(dadosFrame)

# verifica os tipos dos dados
print(dadosFrame.dtypes) # mostra os tipos de cada coluna

# Converter os tipos de dados das variavéis
for col in dadosFrame.columns:
    if col != 'AG05':
        dadosFrame[col] = pd.to_numeric(dadosFrame[col], errors="coerce").fillna(0).astype(int)

# Converter AG05 para categoria (se for o caso)
dados['AG05'] = dadosFrame['AG05'].astype('category')
print(dadosFrame.dtypes)

############# Exibindo os atributos numéricos #################

# count: Número de valores não nulos na coluna (exclui NaN)
# mean:  Média aritmética dos valores
# std:   Desvio padrão (medida da dispersão dos dados em relação à média)
# min:   Menor valor da coluna
# 25%:  Primeiro quartil (25% dos dados estão abaixo desse valor)
# 50%:   Mediana (metade dos dados estão abaixo desse valor)
# 75%:   Terceiro quartil (75% dos dados estão abaixo desse valor)
# max:   Maior valor da coluna

print(dadosFrame.select_dtypes('number').describe())
print(dadosFrame.select_dtypes)

# carregar dframe em dados
data = dadosFrame
# Resetar os índices
data.reset_index(drop=True, inplace=True)

# Ridge Classifier
# X = variáveis, y = target (alvo)

X = data.drop(columns=['AG05'])
y = data['AG05']

# separa treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# padroniza os dados (importante para Ridge)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# cria o modelo Ridge Classifier
ridge_clf = RidgeClassifier(alpha=1.0)  # alpha controla a regularização
ridge_clf.fit(X_train_scaled, y_train)

# previsões
y_pred = ridge_clf.predict(X_test_scaled)

# métricas
print("Acurácia:", accuracy_score(y_test, y_pred))

# pega os coeficientes treinados

coefs = ridge_clf.coef_[0]  # [0] caso seja classificação binária
features = X.columns

# cria um DataFrame para organizar
coef_df = pd.DataFrame({
    "Variável": features,
    "Coeficiente": coefs,
    "Importância (abs)": np.abs(coefs)
})

# ordena pela importância
coef_df = coef_df.sort_values(by="Importância (abs)", ascending=False)

print("Variável mais relevante:", coef_df.iloc[0]["Variável"])
# display(coef_df)

# ordena pelos coeficientes

coef_signed = coef_df.sort_values(by="Coeficiente", ascending=True)

colors = ["red" if c < 0 else "blue" for c in coef_signed["Coeficiente"]]

plt.figure(figsize=(8,6))
plt.barh(coef_signed["Variável"], coef_signed["Coeficiente"], color=colors)
plt.axvline(0, color="black", linewidth=1)  # linha de referência no zero
plt.xlabel("Coeficiente")
plt.ylabel("Variáveis")
plt.title("Coeficientes do Ridge Classifier (Positivo x Negativo)")
plt.tight_layout()
plt.show()
