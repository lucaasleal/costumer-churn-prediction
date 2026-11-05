from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

df = pd.read_excel('base_churn_5000_VF2_aluno_preprocessada.xlsx')

#Retirada de id e churn pelo viés 
#DATA_AQUISICAO_CLIENTE deve virar tempo de contrato e DATA_OCORRENCIA_CHURN é para análise de 30, 60, 90 dias.
X = df.drop(columns=['id_cliente', 'churn'])
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

# Seleciona as colunas originais que são numéricas e podem conter nulos
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

imputer = KNNImputer(n_neighbors=5, metric='nan_euclidean')

# Regra de Ouro: Aprende (fit) APENAS no Treino para não ter Data Leakage
X_train_final[numeric_features] = imputer.fit_transform(X_train_final[numeric_features])

# Aplica as regras aprendidas (transform) no Teste
X_test_final[numeric_features] = imputer.transform(X_test_final[numeric_features])


# Treinando o modelo de árvore de decisão
model = DecisionTreeClassifier(random_state=42, max_depth=5)
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

os.makedirs('results/cf', exist_ok=True)  # Cria o diretório 'results/matrizes_confusao' se não existir

# Cria uma figura exclusiva para a matriz
fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
disp.plot(cmap='Blues', ax=ax_cm)
plt.title('Matriz de Confusão - Árvore de Decisão')
plt.savefig('results/cf/arvore_decisao.png', dpi=300, bbox_inches='tight')
plt.close(fig_cm) # FECHA a figura da matriz para não interferir na próxima
# Define um tamanho grande para a figura, pois árvores de decisão ocupam muito espaço

fig_tree = plt.figure(figsize=(25, 12))

# Plota a estrutura da árvore
plot_tree(
    model,
    feature_names=X_train_final.columns, # Mostra os nomes das variáveis nas perguntas
    class_names=['Não Churn', 'Churn'],  # Mostra a classe predominante no nó
    filled=True,                         # Colore os nós de acordo com a classe
    rounded=True,                        # Arredonda as bordas das caixas
    fontsize=10,
    max_depth=3                          # Limita a exibição a 3 níveis para não ficar ilegível
)

plt.title('Estrutura da Árvore de Decisão')
plt.savefig('results/arvore_decisao_estrutura.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# 3. Importância das Variáveis (Feature Importance)
# ==========================================
# Extrai os valores de importância calculados pelo modelo e os nomes das colunas
importancias = model.feature_importances_
nomes_features = X_train_final.columns

# Cria um DataFrame para organizar e ordenar os dados
df_importancia = pd.DataFrame({
    'Feature': nomes_features,
    'Importancia': importancias
})

# Ordena da variável mais importante para a menos importante
df_importancia = df_importancia.sort_values(by='Importancia', ascending=False)

# Exibe no terminal apenas as variáveis que o modelo realmente usou (importância > 0)
print("\n--- Importância das Variáveis ---")
print(df_importancia[df_importancia['Importancia'] > 0])

# Criação do gráfico de barras horizontais
fig_feat, ax_feat = plt.subplots(figsize=(10, 6))

# Seleciona apenas as top 10 features para o gráfico não ficar poluído
top_features = df_importancia.head(10)

ax_feat.barh(top_features['Feature'], top_features['Importancia'], color='steelblue')
ax_feat.invert_yaxis() # Coloca a feature mais importante no topo do gráfico
ax_feat.set_xlabel('Importância (Gini / Redução de Impureza)')
ax_feat.set_title('Top 10 Features Mais Importantes - Árvore de Decisão')

# Salva a imagem em uma nova pasta dentro de results
os.makedirs('results/features', exist_ok=True)
plt.savefig('results/features/importancia_variaveis.png', dpi=300, bbox_inches='tight')
plt.close(fig_feat)