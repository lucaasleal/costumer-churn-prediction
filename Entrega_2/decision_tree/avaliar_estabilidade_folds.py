"""Diagnóstico de estabilidade por fold para modelos de churn já otimizados.

Usar com X_train / y_train ORIGINAIS (70% de treinamento antes do KNN e OHE).
Mantém os hiperparâmetros escolhidos pelo Optuna; não faz novas buscas.
O ajuste do pré-processamento acontece *dentro* de cada fold.

Atenção: se os mesmos folds foram usados para escolher os hiperparâmetros,
esta análise é diagnóstica, não uma avaliação independente do processo de seleção.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix,
    f1_score, fbeta_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


def avaliar_estabilidade_folds(
    X_train,
    y_train,
    candidatos,
    *,
    n_splits=5,
    pasta_results="results",
    precisao_referencia=0.50,
):
    """Avalia os melhores parâmetros *já escolhidos* em folds individuais.

    candidatos = {
        "nome": {"estudo": estudo_optuna, "limiar": 0.24},
        ...
    }

    Retorna: (df_folds, df_resumo).
    """
    if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.Series):
        raise TypeError("Passe X_train (DataFrame) e y_train (Series), antes do pré-processamento.")
    if len(X_train) != len(y_train):
        raise ValueError("X_train e y_train precisam ter o mesmo número de linhas.")
    if not candidatos:
        raise ValueError("Informe ao menos um candidato.")

    categoricas = ["tipo_contrato", "forma_pagamento"]
    numericas = X_train.select_dtypes(include=[np.number]).columns.tolist()
    sem_tratamento = set(X_train.columns) - set(categoricas) - set(numericas)
    if sem_tratamento:
        raise ValueError(f"Colunas sem pré-processamento definido: {sorted(sem_tratamento)}")
    if not set(categoricas).issubset(X_train.columns):
        raise ValueError("Ausência de uma das colunas categóricas esperadas.")
    if not {0, 1}.issubset(set(y_train.unique())):
        raise ValueError("y_train deve conter as classes 0 e 1, sendo 1 = churn.")

    folds = list(
        StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42,
        ).split(X_train, y_train)
    )
    pasta = Path(pasta_results)
    pasta.mkdir(parents=True, exist_ok=True)
    historico = []

    for nome, config in candidatos.items():
        estudo = config["estudo"]
        limiar = float(config["limiar"])
        if not 0 < limiar < 1:
            raise ValueError(f"Limiar inválido para {nome}: {limiar}")
        # Checagens para não avaliar inadvertidamente uma condição diferente da otimizada.
        limiar_estudo = estudo.user_attrs.get("limiar")
        if limiar_estudo is not None and not np.isclose(limiar, limiar_estudo):
            raise ValueError(f"{nome}: limiar {limiar} diferente do estudo ({limiar_estudo}).")
        folds_estudo = estudo.user_attrs.get("n_splits")
        if folds_estudo is not None and n_splits != int(folds_estudo):
            raise ValueError(f"{nome}: Optuna usou {folds_estudo} folds, não {n_splits}.")
        melhor_trial = estudo.best_trial
        if melhor_trial.user_attrs.get("elegivel") is False:
            raise ValueError(f"{nome}: melhor trial não satisfez a restrição de precisão.")
        params = melhor_trial.user_attrs["parametros_arvore"].copy()

        for numero_fold, (idx_treino, idx_val) in enumerate(folds, start=1):
            X_tr, X_val = X_train.iloc[idx_treino], X_train.iloc[idx_val]
            y_tr, y_val = y_train.iloc[idx_treino], y_train.iloc[idx_val]
            preproc = ColumnTransformer(
                transformers=[
                    ("numericas", KNNImputer(n_neighbors=5, metric="nan_euclidean"), numericas),
                    ("categoricas", OneHotEncoder(handle_unknown="ignore"), categoricas),
                ],
                sparse_threshold=0.0,
            )
            modelo = Pipeline([
                ("preprocessamento", preproc),
                ("arvore", DecisionTreeClassifier(random_state=42, **params)),
            ])
            modelo.fit(X_tr, y_tr)
            classe_idx = list(modelo.named_steps["arvore"].classes_).index(1)
            prob_churn = modelo.predict_proba(X_val)[:, classe_idx]
            y_pred = (prob_churn >= limiar).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()

            historico.append({
                "modelo": nome,
                "fold": numero_fold,
                "limiar": limiar,
                "n_validacao": len(y_val),
                "churn_reais": int((y_val == 1).sum()),
                "churn_previstos": int(y_pred.sum()),
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
                "recall": recall_score(y_val, y_pred, pos_label=1, zero_division=0),
                "precisao": precision_score(y_val, y_pred, pos_label=1, zero_division=0),
                "f1": f1_score(y_val, y_pred, pos_label=1, zero_division=0),
                "f2": fbeta_score(y_val, y_pred, beta=2, pos_label=1, zero_division=0),
                "acuracia": accuracy_score(y_val, y_pred),
                "kappa": cohen_kappa_score(y_val, y_pred),
                "profundidade": modelo.named_steps["arvore"].get_depth(),
                "numero_folhas": modelo.named_steps["arvore"].get_n_leaves(),
            })

    df_folds = pd.DataFrame(historico)
    df_folds.to_csv(pasta / "estabilidade_folds_detalhes.csv", index=False)

    resumo = []
    for nome, g in df_folds.groupby("modelo", sort=False):
        linha = {"modelo": nome, "limiar": float(g["limiar"].iloc[0])}
        for metrica in ("recall", "precisao", "f1", "f2", "acuracia", "kappa"):
            linha[f"{metrica}_media"] = g[metrica].mean()
            linha[f"{metrica}_dp"] = g[metrica].std(ddof=1)
        linha["precisao_min_fold"] = g["precisao"].min()
        linha["precisao_max_fold"] = g["precisao"].max()
        linha["folds_precisao_ge_referencia"] = int((g["precisao"] >= precisao_referencia).sum())
        linha["total_folds"] = len(g)
        resumo.append(linha)
    df_resumo = pd.DataFrame(resumo)
    df_resumo.to_csv(pasta / "estabilidade_folds_resumo.csv", index=False)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for nome, g in df_folds.groupby("modelo", sort=False):
        axs[0].plot(g["fold"], g["recall"], marker="o", label=nome)
        axs[1].plot(g["fold"], g["precisao"], marker="o", label=nome)
    axs[0].set_title("Recall de churn por fold")
    axs[1].set_title("Precisão de churn por fold")
    axs[1].axhline(precisao_referencia, linestyle="--", color="gray", label=f"Referência: {precisao_referencia:.0%}")
    for ax in axs:
        ax.set_xlabel("Fold")
        ax.set_ylabel("Pontuação")
        ax.set_xticks(range(1, n_splits + 1))
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Estabilidade dos candidatos (treinamento 70%, validação cruzada)")
    fig.tight_layout()
    fig.savefig(pasta / "estabilidade_folds.png", dpi=180, bbox_inches="tight")
    plt.show()

    return df_folds, df_resumo
