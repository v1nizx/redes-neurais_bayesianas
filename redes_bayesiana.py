import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 🔹 1. Carregar o dataset
arquivo = "G:/MARCOS VINICIUS/Python/redes Bayesiana/content/diabetes.csv"
colunas = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
           "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

# Ler o CSV
data = pd.read_csv(arquivo, names=colunas, skiprows=1)

# 🔹 2. Pré-processamento
# Substituir valores zero em colunas críticas por NaN
colunas_para_corrigir = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
data[colunas_para_corrigir] = data[colunas_para_corrigir].replace(0, np.nan)

# Preencher valores NaN com a média da coluna
data.fillna(data.mean(), inplace=True)

# 🔹 3. Separar Features (X) e Target (y)
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# 🔹 4. Dividir em Conjuntos de Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 5. Criar e Treinar o Modelo Naïve Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# 🔹 6. Fazer Previsões
y_pred = model.predict(X_test)

# 🔹 7. Avaliar o Modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy:.2%}\n")
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))
print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred))

# 🔹 8. Exibir Importância das Features (aproximada pela média dos valores)
feature_importance = np.abs(model.theta_[1] - model.theta_[0])  # Diferença nas médias entre classes
plt.barh(X.columns, feature_importance)
plt.xlabel("Importância da Feature")
plt.title("Importância das Features no Modelo Bayesiano")
plt.show()
