from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

df = pd.read_excel('base_churn_5000_VF2_aluno_preprocessada.xlsx')

#Retirada de id e churn pelo viés 
#DATA_AQUISICAO_CLIENTE deve virar tempo de contrato e DATA_OCORRENCIA_CHURN é para análise de 30, 60, 90 dias.
X = df.drop(columns=['id_cliente', 'churn', 'data_aquisicao_cliente', 'data_ocorrencia_churn'])
y = df['churn']


#Dividindo os dados em treino e teste, com 30% dos dados para teste e estratificando pelo target churn
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# One-hot encoding para variáveis categóricas tipo_contrato e forma_pagamento
categorical_features = ['tipo_contrato', 'forma_pagamento']
encoder = OneHotEncoder(handle_unknown='ignore')
X_train_encoded = encoder.fit_transform(X_train[categorical_features])
X_test_encoded = encoder.transform(X_test[categorical_features])


# Criando DataFrames para os dados codificados
df_encoded_train = pd.DataFrame(X_train_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_features), index=X_train.index)
df_encoded_test = pd.DataFrame(X_test_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_features), index=X_test.index)


#Concatenando os dados codificados com os dados originais, removendo as colunas categóricas originais
X_train_final = pd.concat([
    X_train.drop(columns=categorical_features),
    df_encoded_train
], axis=1)

X_test_final = pd.concat([
    X_test.drop(columns=categorical_features),
    df_encoded_test
], axis=1)


# Treinando o modelo de árvore de decisão
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train_final, y_train)

y_pred = model.predict(X_test_final)

# Calculando as métricas
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print(f"Acurácia: {accuracy:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"Recall: {recall:.2f}")
print(f"Precision: {precision:.2f}")

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Não Churn', 'Churn']
)

os.makedirs('results', exist_ok=True)  # Cria o diretório 'results' se não existir
os.makedirs('results/cf', exist_ok=True)  # Cria o diretório 'results/matrizes_confusao' se não existir
disp.plot(cmap='Blues')
plt.title('Matriz de Confusão - Árvore de Decisão')
plt.savefig('results/cf/arvore_decisao.png')  # Salva a figura como arquivo PNG
plt.show()