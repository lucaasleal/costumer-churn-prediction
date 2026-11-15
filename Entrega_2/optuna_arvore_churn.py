"""Comparações de árvores para churn: estudos A/B/C, com e sem pós-poda.

Mantém KNNImputer e OneHotEncoder dentro dos folds de validação cruzada.
Os estudos são persistidos em SQLite na pasta results.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    fbeta_score, cohen_kappa_score,
)


def _storage(pasta_results):
    pasta = Path(pasta_results)
    pasta.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(pasta / 'estudos_optuna.db').resolve().as_posix()}"


def _pipeline(params, numericas, categoricas):
    preproc = ColumnTransformer([
        ('numericas', KNNImputer(n_neighbors=5, metric='nan_euclidean'), numericas),
        ('categoricas', OneHotEncoder(handle_unknown='ignore'), categoricas),
    ], sparse_threshold=0.0)
    return Pipeline([
        ('preprocessamento', preproc),
        ('arvore', DecisionTreeClassifier(random_state=42, **params)),
    ])


def _sugerir_parametros(trial, com_pos_poda):
    limitar_folhas = trial.suggest_categorical('limitar_folhas', [False, True])
    usar_impurity = trial.suggest_categorical('usar_min_impurity_decrease', [False, True])
    return {
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'max_depth': trial.suggest_int('max_depth', 2, 25),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 150),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 100),
        'max_leaf_nodes': (
            trial.suggest_int('max_leaf_nodes', 2, 250) if limitar_folhas else None
        ),
        'min_impurity_decrease': (
            trial.suggest_float('min_impurity_decrease_positivo', 1e-5, 1e-2, log=True)
            if usar_impurity else 0.0
        ),
        'ccp_alpha': (
            trial.suggest_float('ccp_alpha', 1e-5, 1e-2, log=True)
            if com_pos_poda else 0.0
        ),
    }


def _scoring_para_limiar(limiar):
    """Scorers calculados pelas probabilidades da classe churn (rótulo 1)."""
    def scorer(metrica):
        def aplicar(modelo, X_val, y_val):
            classes = list(modelo.classes_)
            indice_churn = classes.index(1)
            prob = modelo.predict_proba(X_val)[:, indice_churn]
            y_pred = (prob >= limiar).astype(int)
            if metrica == 'recall_churn':
                return recall_score(y_val, y_pred, pos_label=1, zero_division=0)
            if metrica == 'precisao_churn':
                return precision_score(y_val, y_pred, pos_label=1, zero_division=0)
            if metrica == 'f1_churn':
                return f1_score(y_val, y_pred, pos_label=1, zero_division=0)
            if metrica == 'f2_churn':
                return fbeta_score(y_val, y_pred, beta=2, pos_label=1, zero_division=0)
            if metrica == 'acuracia':
                return accuracy_score(y_val, y_pred)
            if metrica == 'kappa':
                return cohen_kappa_score(y_val, y_pred)
            raise ValueError(metrica)
        return aplicar
    return {nome: scorer(nome) for nome in (
        'recall_churn', 'precisao_churn', 'f1_churn',
        'f2_churn', 'acuracia', 'kappa'
    )}


def executar_experimento(
    X_train, y_train, *, nome_experimento, limiar,
    objetivo='recall_churn', precisao_minima=None,
    n_trials=60, n_splits=5, pasta_results='results',
    continuar=False,
):
    """Roda/compara um experimento com duas variantes (sem e com pós-poda).

    objetivo: 'recall_churn' ou 'f2_churn'.
    continuar=False: reutiliza estudos existentes sem adicionar novas tentativas.
    continuar=True: acrescenta n_trials a cada estudo existente.
    """
    if objetivo not in ('recall_churn', 'f2_churn'):
        raise ValueError("objetivo deve ser 'recall_churn' ou 'f2_churn'")
    if not 0 < limiar < 1:
        raise ValueError('limiar deve estar estritamente entre 0 e 1')
    if precisao_minima is not None and not 0 <= precisao_minima <= 1:
        raise ValueError('precisao_minima deve estar entre 0 e 1')
    if n_splits < 2:
        raise ValueError('n_splits deve ser >= 2')

    categoricas = ['tipo_contrato', 'forma_pagamento']
    numericas = X_train.select_dtypes(include=[np.number]).columns.tolist()
    colunas_sem_tratamento = set(X_train.columns) - set(categoricas) - set(numericas)
    if colunas_sem_tratamento:
        raise ValueError(f'Colunas sem tratamento: {colunas_sem_tratamento}')

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = _scoring_para_limiar(limiar)
    pasta = Path(pasta_results)
    storage = _storage(pasta)

    def objective(trial, com_pos_poda):
        params = _sugerir_parametros(trial, com_pos_poda)
        resultados = cross_validate(
            _pipeline(params, numericas, categoricas), X_train, y_train,
            cv=cv, scoring=scoring, n_jobs=1, error_score='raise',
        )
        metricas = {
            nome: float(np.mean(resultados[f'test_{nome}']))
            for nome in scoring
        }
        for nome, valor in metricas.items():
            trial.set_user_attr(nome, valor)
        trial.set_user_attr('parametros_arvore', params)
        elegivel = (precisao_minima is None or
                    metricas['precisao_churn'] >= precisao_minima)
        trial.set_user_attr('elegivel', elegivel)
        # Penalidade apenas para escolher a melhor tentativa; métricas reais
        # continuam preservadas em user_attrs.
        return metricas[objetivo] if elegivel else -1.0

    estudos = {}
    resumo = []
    for sufixo, com_pos_poda in [('sem_pos_poda', False), ('com_pos_poda', True)]:
        nome = f'{nome_experimento}_{sufixo}'
        estudo = optuna.create_study(
            direction='maximize', study_name=nome, storage=storage,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        configuracao = {
            'limiar': float(limiar), 'objetivo': objetivo,
            'precisao_minima': precisao_minima,
            'n_splits': int(n_splits), 'com_pos_poda': com_pos_poda,
        }
        if estudo.trials:
            for chave, valor in configuracao.items():
                if estudo.user_attrs.get(chave) != valor:
                    raise ValueError(
                        f'Estudo {nome} já existe com configuração diferente '
                        f'para {chave}: {estudo.user_attrs.get(chave)} != {valor}. '
                        'Use outro nome_experimento.'
                    )
        else:
            for chave, valor in configuracao.items():
                estudo.set_user_attr(chave, valor)

        if not estudo.trials or continuar:
            estudo.optimize(
                lambda trial: objective(trial, com_pos_poda),
                n_trials=n_trials,
            )
        else:
            print(f'[{nome}] estudo já existente: {len(estudo.trials)} tentativas; '
                  'nenhuma tentativa nova (continuar=False).')

        estudo.trials_dataframe().to_csv(
            pasta / f'optuna_{nome}.csv', index=False
        )
        elegiveis = [
            t for t in estudo.trials
            if t.value is not None and t.user_attrs.get('elegivel', False)
        ]
        if not elegiveis:
            print(f'[{nome}] NENHUMA configuração satisfez a restrição de precisão.')
            resumo.append({'estudo': nome, 'elegivel': False,
                           'tentativas': len(estudo.trials)})
        else:
            melhor = max(elegiveis, key=lambda t: t.value)
            r = {'estudo': nome, 'elegivel': True,
                 'tentativas': len(estudo.trials),
                 'objetivo': objetivo, 'limiar': limiar,
                 'precisao_minima': precisao_minima,
                 'melhor_trial': melhor.number,
                 'melhor_valor_objetivo': melhor.value}
            r.update({k: melhor.user_attrs[k] for k in scoring})
            r['parametros_arvore'] = str(melhor.user_attrs['parametros_arvore'])
            resumo.append(r)
            print(f'[{nome}] {objetivo}={melhor.value:.4f} | '
                  f'recall={r["recall_churn"]:.4f} | '
                  f'precisão={r["precisao_churn"]:.4f} | '
                  f'F2={r["f2_churn"]:.4f} | tentativas={len(estudo.trials)}')
        estudos[sufixo] = estudo

    df_resumo = pd.DataFrame(resumo)
    df_resumo.to_csv(pasta / f'resumo_{nome_experimento}.csv', index=False)
    return estudos, df_resumo


def salvar_estudos_anteriores(estudos_anteriores, pasta_results='results'):
    """Copia estudos antigos que ainda estejam em memória para o banco SQLite.

    Não sobrescreve estudos já copiados.
    """
    storage = _storage(pasta_results)
    copiados = {}
    for nome_antigo, antigo in estudos_anteriores.items():
        nome_novo = f'historico_original_{nome_antigo}'
        novo = optuna.create_study(
            direction='maximize', study_name=nome_novo,
            storage=storage, load_if_exists=True
        )
        if not novo.trials:
            novo.add_trials(antigo.get_trials(deepcopy=True))
        copiados[nome_antigo] = novo
        print(f'{nome_novo}: {len(novo.trials)} tentativas salvas no SQLite.')
    return copiados
