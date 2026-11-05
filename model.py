from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import accuracy_score

from evaluation_utils import (
    calcular_metricas_completas,
    salvar_matriz_confusao,
    exportar_relatorios,
    plotar_painel_curvas
)

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

resultados_experimentos = []

# =========================================================================
# EXPERIMENTO 1: Árvore Baseline Desbalanceada (max_depth=5, limiar=0.50)
# =========================================================================
model = DecisionTreeClassifier(random_state=42, max_depth=5)
model.fit(X_train_final, y_train)

y_prob = model.predict_proba(X_test_final)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

res_base = calcular_metricas_completas(
    "1_Baseline_Desbalanceada_Limiar_0.50",
    y_test, y_pred, y_prob,
    hiperparametros={"max_depth": 5, "criterion": "gini", "balanceamento": "Nenhum"},
    limiar=0.50
)

resultados_experimentos.append(res_base)

# Calculando as métricas
accuracy = accuracy_score(y_test, y_pred)
salvar_matriz_confusao(y_test, y_pred, "Matriz de Confusão - Baseline (Limiar 0.50)", "results/cf/cm_1_baseline.png")

# =========================================================================
# EXPERIMENTO 2: Ajuste de Limiar Operacional (limiar=0.30 para elevar Recall)
# =========================================================================
y_pred_limiar30 = (y_prob >= 0.30).astype(int)
res_limiar30 = calcular_metricas_completas(
    "2_Baseline_Desbalanceada_Limiar_0.30",
    y_test, y_pred_limiar30, y_prob,
    hiperparametros={"max_depth": 5, "criterion": "gini", "balanceamento": "Nenhum"},
    limiar=0.30
)
resultados_experimentos.append(res_limiar30)
salvar_matriz_confusao(y_test, y_pred_limiar30, "Matriz de Confusão - Limiar 0.30", "results/cf/cm_2_limiar_030.png")

# =========================================================================
# EXPERIMENTO 3: Árvore Balanceada com Random Over-Sampling (ROS no Treino)
# =========================================================================
np.random.seed(42)
idx_pos = y_train[y_train == 1].index
idx_neg = y_train[y_train == 0].index
idx_pos_oversampled = np.random.choice(idx_pos, size=len(idx_neg), replace=True)
idx_ros = np.concatenate([idx_neg, idx_pos_oversampled])

X_train_ros = X_train_final.loc[idx_ros]
y_train_ros = y_train.loc[idx_ros]

model_ros = DecisionTreeClassifier(random_state=42, max_depth=5)
model_ros.fit(X_train_ros, y_train_ros)
y_prob_ros = model_ros.predict_proba(X_test_final)[:, 1]
y_pred_ros = model_ros.predict(X_test_final)

res_ros = calcular_metricas_completas(
    "3_Arvore_Balanceada_ROS_Treino",
    y_test, y_pred_ros, y_prob_ros,
    hiperparametros={"max_depth": 5, "criterion": "gini", "balanceamento": "ROS no Treino"},
    limiar=0.50
)
resultados_experimentos.append(res_ros)
salvar_matriz_confusao(y_test, y_pred_ros, "Matriz de Confusão - Balanceada (ROS)", "results/cf/cm_3_balanceada_ros.png")

# =========================================================================
# EXPERIMENTO 4: Sensibilidade de Hiperparâmetros (max_depth de 1 a 15)
# =========================================================================
historico_depths = []
for d in range(1, 16):
    clf_d = DecisionTreeClassifier(random_state=42, max_depth=d)
    clf_d.fit(X_train_final, y_train)
    acc_tr = float(accuracy_score(y_train, clf_d.predict(X_train_final)))
    y_pr_te = clf_d.predict(X_test_final)
    y_pb_te = clf_d.predict_proba(X_test_final)[:, 1]

    res_d = calcular_metricas_completas(
        f"Exp_MaxDepth_{d:02d}",
        y_test, y_pr_te, y_pb_te,
        hiperparametros={"max_depth": d, "criterion": "gini"}
    )
    res_d["metricas"]["acuracia_treino"] = round(acc_tr, 4)
    res_d["metricas"]["gap_overfitting"] = round(acc_tr - res_d["metricas"]["acuracia"], 4)
    historico_depths.append(res_d)

# =========================================================================
# 5. Plot da Estrutura da Árvore e Importância das Variáveis
# =========================================================================
fig_tree = plt.figure(figsize=(25, 12))
plot_tree(
    model,
    feature_names=X_train_final.columns,
    class_names=['Permanência', 'Churn'],
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=3
)
plt.title('Estrutura da Árvore de Decisão (Top 3 Níveis)')
plt.savefig('results/arvore_decisao_estrutura.png', dpi=300, bbox_inches='tight')
plt.close(fig_tree)

df_importancia = pd.DataFrame({
    'Feature': X_train_final.columns,
    'Importancia': model.feature_importances_
}).sort_values(by='Importancia', ascending=False)

fig_feat, ax_feat = plt.subplots(figsize=(10, 6))
top_features = df_importancia.head(10)
ax_feat.barh(top_features['Feature'], top_features['Importancia'], color='steelblue')
ax_feat.invert_yaxis()
ax_feat.set_xlabel('Importância (Gini / Redução de Impureza)')
ax_feat.set_title('Top 10 Features Mais Importantes - Árvore de Decisão')
os.makedirs('results/features', exist_ok=True)
plt.savefig('results/features/importancia_variaveis.png', dpi=300, bbox_inches='tight')
plt.close(fig_feat)

# =========================================================================
# 6. Exportação dos Relatórios (.json e .txt) e Painel de Curvas da Aula 3
# =========================================================================
exportar_relatorios(
    resultados_experimentos,
    historico_depths,
    df_importancia,
    n_total=len(df),
    n_treino=len(X_train_final),
    n_teste=len(X_test_final)
)
plotar_painel_curvas(y_test, y_prob, res_base, historico_depths)
print("Relatórios salvos em 'results/relatorios/' e gráficos salvos em 'results/'!")