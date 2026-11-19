"""Arvore CART com categorias nativas (scikit-learn >= 1.10 dev), sem One-Hot.

Usar este modulo junto com main_arvore_categorica.ipynb. A preparacao do
KNN e sempre ajustada apenas no subconjunto de treinamento (fold) e entrega
um DataFrame pandas com dtype 'category', necessario para a arvore nativa.

Comparacoes Optuna independentes: pre_poda (ccp_alpha=0), pos_poda
(demais parametros de poda no padrao) e hibrida (pre + pos).
"""

from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
import optuna
import sklearn

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    fbeta_score, cohen_kappa_score,
)


CATEGORICAS = ("tipo_contrato", "forma_pagamento")
PREPROCESSAMENTO = "categorias_nativas_knn_v1"


def verificar_suporte_categorico():
    """Falha cedo caso o kernel nao use a build com suporte nativo."""
    if "categorical_features" not in DecisionTreeClassifier().get_params():
        raise RuntimeError(
            "Este notebook requer scikit-learn 1.10 de desenvolvimento "
            "com DecisionTreeClassifier(categorical_features=...). "
            f"Versao do kernel atual: {sklearn.__version__}. "
            "Selecione o kernel Conda churn_cat no Jupyter."
        )
    return sklearn.__version__


class PreparadorArvoreCategorica(TransformerMixin, BaseEstimator):
    """KNN apenas em numericas; conserva texto e dtype category nas demais.

    Ao transformar validacao/teste, categorias nao presentes no treino
    convertem-se em ausentes (NaN), conforme comportamento documentado
    para categorias desconhecidas da arvore nativa.
    """

    def __init__(self, categoricas=CATEGORICAS, n_neighbors=5):
        self.categoricas = categoricas
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X deve ser um DataFrame do pandas.")
        self.colunas_ = list(X.columns)
        self.categoricas_ = list(self.categoricas)
        faltantes = set(self.categoricas_) - set(X.columns)
        if faltantes:
            raise ValueError(f"Colunas categoricas nao encontradas: {faltantes}")
        self.numericas_ = [c for c in self.colunas_ if c not in self.categoricas_]
        inadequadas = [c for c in self.numericas_ if not pd.api.types.is_numeric_dtype(X[c])]
        if inadequadas:
            raise ValueError(f"Colunas nao numericas fora da lista categorica: {inadequadas}")
        self.categorias_ = {}
        for col in self.categoricas_:
            valores = list(pd.Series(X[col].dropna()).unique())
            if len(valores) > 255:
                raise ValueError(f"'{col}' excede 255 categorias distintas no treino")
            self.categorias_[col] = valores
        self.imputador_ = None
        if self.numericas_:
            self.imputador_ = KNNImputer(
                n_neighbors=self.n_neighbors,
                metric="nan_euclidean",
                keep_empty_features=True,
            )
            self.imputador_.fit(X[self.numericas_])
        self.n_features_in_ = len(self.colunas_)
        self.feature_names_in_ = np.asarray(self.colunas_, dtype=object)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X deve ser um DataFrame do pandas.")
        faltantes = set(self.colunas_) - set(X.columns)
        if faltantes:
            raise ValueError(f"Colunas do treinamento ausentes: {faltantes}")
        saida = pd.DataFrame(index=X.index)
        if self.numericas_:
            numeros = self.imputador_.transform(X[self.numericas_])
            for j, col in enumerate(self.numericas_):
                saida[col] = numeros[:, j]
        for col in self.categoricas_:
            # Converte categoria nao vista em NaN, sem consultar y.
            saida[col] = pd.Categorical(
                X[col].astype(object), categories=self.categorias_[col]
            )
        return saida[self.colunas_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.colunas_, dtype=object)


def criar_preprocessador(categoricas=CATEGORICAS):
    return PreparadorArvoreCategorica(categoricas=categoricas, n_neighbors=5)


def criar_pipeline(parametros=None, *, categoricas=CATEGORICAS):
    verificar_suporte_categorico()
    params = dict(parametros or {})
    params.pop("random_state", None)
    params.pop("categorical_features", None)
    return Pipeline([
        ("preprocessamento", criar_preprocessador(categoricas)),
        ("arvore", DecisionTreeClassifier(
            categorical_features="from_dtype",
            random_state=42,
            **params,
        )),
    ])


def parametros_baseline():
    """Arvore sem poda explicita (criterion e splitter fixos)."""
    return {"criterion": "gini", "splitter": "best", "ccp_alpha": 0.0}


def _sugerir_pre_poda(trial):
    limitar_folhas = trial.suggest_categorical("limitar_folhas", [False, True])
    usar_impurity = trial.suggest_categorical("usar_min_impurity_decrease", [False, True])
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 120),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 80),
        "max_leaf_nodes": (
            trial.suggest_int("max_leaf_nodes", 2, 200) if limitar_folhas else None
        ),
        "min_impurity_decrease": (
            trial.suggest_float("min_impurity_decrease", 1e-6, 1e-2, log=True)
            if usar_impurity else 0.0
        ),
    }


def sugerir_parametros(trial, estrategia):
    """Espacos de busca EXCLUSIVOS para comparacao justa de regularizacao."""
    if estrategia not in ("pre_poda", "pos_poda", "hibrida"):
        raise ValueError(estrategia)
    params = parametros_baseline()
    if estrategia in ("pre_poda", "hibrida"):
        params.update(_sugerir_pre_poda(trial))
    if estrategia in ("pos_poda", "hibrida"):
        params["ccp_alpha"] = trial.suggest_float(
            "ccp_alpha", 1e-6, 3e-2, log=True
        )
    return params


def _scoring_para_limiar(limiar):
    def criar_scorer(nome):
        def scorer(pipeline, X_val, y_val):
            classes = list(pipeline.named_steps["arvore"].classes_)
            probs = pipeline.predict_proba(X_val)[:, classes.index(1)]
            pred = (probs >= limiar).astype(int)
            if nome == "recall_churn":
                return recall_score(y_val, pred, zero_division=0)
            if nome == "precisao_churn":
                return precision_score(y_val, pred, zero_division=0)
            if nome == "f1_churn":
                return f1_score(y_val, pred, zero_division=0)
            if nome == "f2_churn":
                return fbeta_score(y_val, pred, beta=2, zero_division=0)
            if nome == "acuracia":
                return accuracy_score(y_val, pred)
            if nome == "kappa":
                return cohen_kappa_score(y_val, pred)
            raise ValueError(nome)
        return scorer
    nomes = ("recall_churn", "precisao_churn", "f1_churn",
             "f2_churn", "acuracia", "kappa")
    return {nome: criar_scorer(nome) for nome in nomes}


def _assinatura_dados(X, y):
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(X, index=True).values.tobytes())
    h.update(pd.util.hash_pandas_object(pd.Series(y), index=True).values.tobytes())
    return h.hexdigest()[:16]


def melhor_trial_elegivel(estudo):
    elegiveis = [t for t in estudo.trials
                 if t.value is not None and t.user_attrs.get("elegivel", False)]
    if not elegiveis:
        raise ValueError(
            f"O estudo {estudo.study_name!r} nao possui trials que atendam a precisao."
        )
    return max(elegiveis, key=lambda t: t.value)


def preparar_folds_sem_vazamento(X, y, cv):
    """Preprocessa cada fold UMA VEZ e reutiliza em todos os trials.

    O imputador e ajustado apenas com o treino do respectivo fold.
    Como a preparacao nao usa y, reusa-la e seguro e acelera o Optuna.
    """
    folds = []
    for ind_tr, ind_val in cv.split(X, y):
        X_tr, X_val = X.iloc[ind_tr], X.iloc[ind_val]
        y_tr, y_val = y.iloc[ind_tr], y.iloc[ind_val]
        pre = criar_preprocessador()
        Xt = pre.fit_transform(X_tr)
        Xv = pre.transform(X_val)
        folds.append((Xt, Xv, y_tr, y_val))
    return folds


def _estatisticas_cv(params, folds, limiar):
    """CV rapida, preservando os mesmos folds/preprocessamento por trial."""
    metricas_folds = []
    for Xt, Xv, yt, yv in folds:
        arvore = DecisionTreeClassifier(
            categorical_features="from_dtype", random_state=42, **params
        )
        arvore.fit(Xt, yt)
        prob = arvore.predict_proba(Xv)[:, list(arvore.classes_).index(1)]
        pred = (prob >= limiar).astype(int)
        metricas_folds.append({
            "recall_churn": recall_score(yv, pred, zero_division=0),
            "precisao_churn": precision_score(yv, pred, zero_division=0),
            "f1_churn": f1_score(yv, pred, zero_division=0),
            "f2_churn": fbeta_score(yv, pred, beta=2, zero_division=0),
            "acuracia": accuracy_score(yv, pred),
            "kappa": cohen_kappa_score(yv, pred),
            "profundidade": arvore.get_depth(),
            "folhas": arvore.get_n_leaves(),
        })
    df = pd.DataFrame(metricas_folds)
    saida = {}
    for nome in df.columns:
        saida[nome] = float(df[nome].mean())
        saida[f"{nome}_dp"] = float(df[nome].std(ddof=1))
    return saida


def executar_comparacao_podas(
    X_train, y_train, *, nome_experimento="principal", limiar=0.30,
    objetivo="f2_churn", precisao_minima=0.50, n_trials=60, n_splits=5,
    pasta_results="results/arvore_categorica_110", continuar=False,
):
    """Baseline e 3 Optunas: pre-poda, pos-poda so ccp_alpha, hibrida.

    Precisao minima compara a MEDIA dos folds (nao garante cada fold).
    A CV e utilizada para selecionar parametros; nao e uma estimativa
    independente de desempenho apos a escolha. Reservar X_test para final.
    """
    verificar_suporte_categorico()
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train deve ser DataFrame pandas.")
    if set(CATEGORICAS) - set(X_train.columns):
        raise ValueError(f"Faltam as categorias: {set(CATEGORICAS)-set(X_train.columns)}")
    if set(pd.Series(y_train).dropna().unique()) != {0, 1}:
        raise ValueError("y_train precisa conter somente as classes 0 e 1.")
    if objetivo not in ("recall_churn", "f2_churn"):
        raise ValueError("objetivo deve ser recall_churn ou f2_churn")
    if not 0 < limiar < 1:
        raise ValueError("limiar deve estar entre 0 e 1, exclusive")
    if precisao_minima is not None and not 0 <= precisao_minima <= 1:
        raise ValueError("precisao_minima fora de [0,1]")
    if n_splits < 2 or n_trials < 1:
        raise ValueError("n_splits >= 2 e n_trials >= 1")

    pasta = Path(pasta_results)
    pasta.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(pasta / 'estudos_optuna_categorico.db').resolve().as_posix()}"
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # Cada fold ajusta KNN somente no proprio treino, uma unica vez.
    folds_prontos = preparar_folds_sem_vazamento(X_train, y_train, cv)
    assinatura = _assinatura_dados(X_train, y_train)
    configuracao = dict(
        preprocessing=PREPROCESSAMENTO,
        sklearn_version=sklearn.__version__,
        dataset_fingerprint=assinatura,
        objetivo=objetivo,
        limiar=float(limiar),
        precisao_minima=(None if precisao_minima is None else float(precisao_minima)),
        n_splits=int(n_splits),
    )

    baseline_cv = _estatisticas_cv(parametros_baseline(), folds_prontos, limiar)
    resumo = [dict(estudo=f"{nome_experimento}_baseline", estrategia="baseline",
                   elegivel=(precisao_minima is None or
                             baseline_cv['precisao_churn'] >= precisao_minima),
                   parametros_arvore=json.dumps(parametros_baseline()),
                   **baseline_cv)]
    estudos = {}

    for estrategia in ("pre_poda", "pos_poda", "hibrida"):
        nome = f"{nome_experimento}_{estrategia}"
        estudo = optuna.create_study(
            study_name=nome, storage=storage, load_if_exists=True,
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
        )
        attrs = dict(configuracao, estrategia=estrategia)
        if len(estudo.trials):
            for k, v in attrs.items():
                if estudo.user_attrs.get(k) != v:
                    raise ValueError(
                        f"O estudo {nome} foi criado com {k}="
                        f"{estudo.user_attrs.get(k)!r}, mas agora vale {v!r}. "
                        "Use outro nome_experimento ou pasta_results."
                    )
        else:
            for k, v in attrs.items():
                estudo.set_user_attr(k, v)

        def objective(trial):
            params = sugerir_parametros(trial, estrategia)
            metricas = _estatisticas_cv(params, folds_prontos, limiar)
            elegivel = (
                precisao_minima is None or
                metricas["precisao_churn"] >= precisao_minima
            )
            trial.set_user_attr("parametros_arvore", params)
            trial.set_user_attr("elegivel", elegivel)
            for k, v in metricas.items():
                trial.set_user_attr(k, v)
            return metricas[objetivo] if elegivel else -1.0

        if continuar or not estudo.trials:
            estudo.optimize(objective, n_trials=n_trials)
        else:
            print(f"{nome}: reutilizando {len(estudo.trials)} trials existentes")
        estudo.trials_dataframe().to_csv(pasta / f"optuna_{nome}.csv", index=False)

        try:
            melhor = melhor_trial_elegivel(estudo)
            linha = dict(
                estudo=nome, estrategia=estrategia, elegivel=True,
                melhor_trial=melhor.number,
                parametros_arvore=json.dumps(
                    melhor.user_attrs["parametros_arvore"], ensure_ascii=False
                ),
            )
            linha.update({k: melhor.user_attrs[k] for k in baseline_cv})
            print(f"{nome}: recall={linha['recall_churn']:.4f}, "
                  f"precisao={linha['precisao_churn']:.4f}, "
                  f"F2={linha['f2_churn']:.4f}")
        except ValueError:
            linha = dict(estudo=nome, estrategia=estrategia, elegivel=False)
            warnings.warn(f"{nome}: nenhum trial com precisao minima exigida")
        resumo.append(linha)
        estudos[estrategia] = estudo

    df_resumo = pd.DataFrame(resumo)
    df_resumo.to_csv(pasta / f"resumo_{nome_experimento}.csv", index=False)
    return estudos, df_resumo


def executar_experimento(
    X_train, y_train, *, nome_experimento, limiar, objetivo="recall_churn",
    precisao_minima=None, n_trials=60, n_splits=5,
    pasta_results="results/arvore_categorica_110", continuar=False,
):
    """Alias para chamadas antigas; retorna as tres estrategias separadas."""
    return executar_comparacao_podas(
        X_train, y_train, nome_experimento=nome_experimento, limiar=limiar,
        objetivo=objetivo, precisao_minima=precisao_minima,
        n_trials=n_trials, n_splits=n_splits,
        pasta_results=pasta_results, continuar=continuar,
    )


def avaliar_estabilidade_folds(
    X_train, y_train, candidatos, *, n_splits=5,
    pasta_results="results/arvore_categorica_110", precisao_referencia=0.50,
):
    """Reavalia as configuracoes selecionadas usando os mesmos folds.

    Diagnostico de variabilidade, NAO teste independente (folds usados Optuna).
    candidatos: {nome: {'parametros': dict, 'limiar': float}}.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    linhas = []
    for nome, item in candidatos.items():
        for fold, (it, iv) in enumerate(cv.split(X_train, y_train), 1):
            X_tr, X_val = X_train.iloc[it], X_train.iloc[iv]
            y_tr, y_val = y_train.iloc[it], y_train.iloc[iv]
            modelo = criar_pipeline(item["parametros"])
            modelo.fit(X_tr, y_tr)
            classe_pos = list(modelo.named_steps["arvore"].classes_).index(1)
            prob = modelo.predict_proba(X_val)[:, classe_pos]
            pred = (prob >= item["limiar"]).astype(int)
            arvore = modelo.named_steps["arvore"]
            linhas.append(dict(
                modelo=nome, fold=fold,
                recall=recall_score(y_val, pred, zero_division=0),
                precisao=precision_score(y_val, pred, zero_division=0),
                f2=fbeta_score(y_val, pred, beta=2, zero_division=0),
                profundidade=arvore.get_depth(), folhas=arvore.get_n_leaves(),
            ))
    detalhe = pd.DataFrame(linhas)
    resumo = detalhe.groupby("modelo", as_index=False).agg(
        recall_media=("recall", "mean"), recall_dp=("recall", "std"),
        precisao_media=("precisao", "mean"), precisao_dp=("precisao", "std"),
        f2_media=("f2", "mean"),
        profundidade_media=("profundidade", "mean"),
        folhas_medias=("folhas", "mean"),
        folds_precisao_ge_referencia=(
            "precisao", lambda s: int((s >= precisao_referencia).sum())
        ),
    )
    pasta = Path(pasta_results)
    pasta.mkdir(parents=True, exist_ok=True)
    detalhe.to_csv(pasta / "estabilidade_folds_detalhes.csv", index=False)
    resumo.to_csv(pasta / "estabilidade_folds_resumo.csv", index=False)
    return detalhe, resumo


def estudar_alpha_fixo(
    X_train, y_train, parametros_pre_poda, *, limiar=0.30,
    alphas=None, n_splits=5, pasta_results="results/arvore_categorica_110",
):
    """Efeito ADICIONAL do ccp_alpha sobre uma pre-poda fixa.

    Este estudo complementar nao substitui a pos-poda isolada do Optuna.
    """
    if alphas is None:
        alphas = np.r_[0.0, np.geomspace(1e-6, 2e-2, 25)]
    params = dict(parametros_pre_poda)
    params.pop("ccp_alpha", None)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    linhas = []
    for fold, (it, iv) in enumerate(cv.split(X_train, y_train), 1):
        X_tr, X_val = X_train.iloc[it], X_train.iloc[iv]
        y_tr, y_val = y_train.iloc[it], y_train.iloc[iv]
        pre = criar_preprocessador()
        Xt = pre.fit_transform(X_tr)
        Xv = pre.transform(X_val)
        for alpha in alphas:
            tree = DecisionTreeClassifier(
                random_state=42, categorical_features="from_dtype",
                **dict(params, ccp_alpha=float(alpha)),
            )
            tree.fit(Xt, y_tr)
            prob = tree.predict_proba(Xv)[:, list(tree.classes_).index(1)]
            pred = (prob >= limiar).astype(int)
            linhas.append(dict(
                fold=fold, ccp_alpha=float(alpha),
                recall=recall_score(y_val, pred, zero_division=0),
                precisao=precision_score(y_val, pred, zero_division=0),
                f2=fbeta_score(y_val, pred, beta=2, zero_division=0),
                acuracia=accuracy_score(y_val, pred),
                nos=tree.tree_.node_count,
                profundidade=tree.get_depth(), folhas=tree.get_n_leaves(),
            ))
    df_folds = pd.DataFrame(linhas)
    df_media = df_folds.groupby("ccp_alpha", as_index=False)[
        ["recall", "precisao", "f2", "acuracia", "nos", "profundidade", "folhas"]
    ].mean()
    pasta = Path(pasta_results)
    pasta.mkdir(parents=True, exist_ok=True)
    df_folds.to_csv(pasta / "ccp_alpha_adicional_folds.csv", index=False)
    df_media.to_csv(pasta / "ccp_alpha_adicional_resumo.csv", index=False)
    return df_folds, df_media
