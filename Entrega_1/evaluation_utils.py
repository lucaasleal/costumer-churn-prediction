import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay
)

def calcular_metricas_completas(nome_exp, y_true, y_pred, y_prob, hiperparametros=None, limiar=0.50):
    """Calcula todas as métricas dos slides (Acurácia, Precisão, Recall, F1, AUROC, KS2 e Matriz de Confusão)."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auroc = float(roc_auc_score(y_true, y_prob))

    # Estatística KS2: separação máxima entre as CDFs de Churn (1) e Permanência (0)
    prob_churn = y_prob[y_true == 1]
    prob_perm = y_prob[y_true == 0]
    ks_res = ks_2samp(prob_churn, prob_perm)
    ks2_stat = float(ks_res.statistic)
    ks2_pval = float(ks_res.pvalue)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]

    return {
        "experimento": nome_exp,
        "hiperparametros": hiperparametros or {},
        "limiar_decisao": float(limiar),
        "metricas": {
            "acuracia": round(acc, 4),
            "precisao": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "auroc": round(auroc, 4),
            "ks2_stat": round(ks2_stat, 4),
            "ks2_p_valor": ks2_pval
        },
        "matriz_confusao": {
            "TN_verdadeiros_negativos": tn,
            "FP_falsos_positivos": fp,
            "FN_falsos_negativos": fn,
            "TP_verdadeiros_positivos": tp,
            "matriz_2x2": cm.tolist()
        }
    }


def salvar_matriz_confusao(y_true, y_pred, titulo, caminho_arquivo):
    """Plota e salva a matriz de confusão individual de um experimento."""
    os.makedirs(os.path.dirname(caminho_arquivo), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Permanência (0)', 'Churn (1)'])
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    ax.set_title(titulo, fontweight='bold')
    plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
    plt.close(fig)


def exportar_relatorios(resultados_principais, historico_depths, df_importancia, n_total, n_treino, n_teste, pasta='./Entrega_1/results/relatorios'):
    """Salva todas as métricas, matrizes de confusão e importâncias em JSON e TXT."""
    os.makedirs(pasta, exist_ok=True)

    dados_json = {
        "dimensionamento_amostra": {
            "total": int(n_total),
            "treino_70_pct": int(n_treino),
            "teste_30_pct": int(n_teste)
        },
        "experimentos_modelos": resultados_principais,
        "sensibilidade_max_depth_1_a_15": historico_depths,
        "importancia_variaveis_gini": df_importancia.round(4).to_dict(orient="records")
    }

    # 1. Exporta JSON
    caminho_json = os.path.join(pasta, 'metricas_modelos.json')
    with open(caminho_json, 'w', encoding='utf-8') as f:
        json.dump(dados_json, f, indent=4, ensure_ascii=False)

    # 2. Exporta TXT formatado
    caminho_txt = os.path.join(pasta, 'relatorio_metricas.txt')
    with open(caminho_txt, 'w', encoding='utf-8') as f:
        f.write("======================================================================\n")
        f.write("   RELATÓRIO DE DESEMPENHO E AUDITORIA DE MODELOS - PREVISÃO CHURN\n")
        f.write("======================================================================\n")
        f.write(f"Total: {n_total} | Treino (70%): {n_treino} | Teste (30%): {n_teste}\n\n")

        for mod in resultados_principais:
            m = mod['metricas']
            cm_d = mod['matriz_confusao']
            f.write("----------------------------------------------------------------------\n")
            f.write(f"EXPERIMENTO: {mod['experimento']}\n")
            f.write(f"Hiperparâmetros: {mod['hiperparametros']} | Limiar: {mod['limiar_decisao']:.2f}\n")
            f.write("----------------------------------------------------------------------\n")
            f.write(f"  - Acurácia  : {m['acuracia']*100:.2f}% ({m['acuracia']:.4f})\n")
            f.write(f"  - Precisão  : {m['precisao']*100:.2f}% ({m['precisao']:.4f})\n")
            f.write(f"  - Recall    : {m['recall']*100:.2f}% ({m['recall']:.4f})\n")
            f.write(f"  - F1-Score  : {m['f1_score']*100:.2f}% ({m['f1_score']:.4f})\n")
            f.write(f"  - AUROC     : {m['auroc']*100:.2f}% ({m['auroc']:.4f})\n")
            f.write(f"  - KS2       : {m['ks2_stat']*100:.2f}% ({m['ks2_stat']:.4f} | p-valor = {m['ks2_p_valor']:.2e})\n")
            f.write("  - Matriz de Confusão (Teste):\n")
            f.write(f"      TN (Permanência correta) = {cm_d['TN_verdadeiros_negativos']} | FP (Alarme falso) = {cm_d['FP_falsos_positivos']}\n")
            f.write(f"      FN (Churn não detectado) = {cm_d['FN_falsos_negativos']} | TP (Churn capturado) = {cm_d['TP_verdadeiros_positivos']}\n\n")

        f.write("======================================================================\n")
        f.write("   SENSIBILIDADE DE HIPERPARÂMETROS (max_depth de 1 a 15)\n")
        f.write("======================================================================\n")
        f.write("Depth | Acc Treino | Acc Teste | Gap Overfit | Recall | F1-Score | AUROC  | KS2\n")
        for item in historico_depths:
            d_val = item['hiperparametros']['max_depth']
            dm = item['metricas']
            f.write(f"  {d_val:02d}  |   {dm['acuracia_treino']:.4f}   |  {dm['acuracia']:.4f}   |   {dm['gap_overfitting']:+.4f}   | {dm['recall']:.4f} |  {dm['f1_score']:.4f}  | {dm['auroc']:.4f} | {dm['ks2_stat']:.4f}\n")

        f.write("\n======================================================================\n")
        f.write("   IMPORTÂNCIA DAS VARIÁVEIS (GINI IMPORTANCE)\n")
        f.write("======================================================================\n")
        for _, row in df_importancia.iterrows():
            f.write(f"  - {row['Feature']:<28}: {row['Importancia']*100:.2f}% ({row['Importancia']:.4f})\n")


def plotar_painel_curvas(y_test, y_prob, res_base, historico_depths, caminho_saida='./Entrega_1/results/curvas/painel_metricas_aula3.png'):
    """Gera o painel com Curva ROC, Teste KS2, Curva Precisão-Recall e Curva de Overfitting (max_depth)."""
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), facecolor='white')

    # 1. Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auroc_val = res_base['metricas']['auroc']
    axes[0, 0].plot(fpr, tpr, color='#1f4e78', lw=2.5, label=f'Curva ROC (AUROC = {auroc_val:.4f})')
    axes[0, 0].plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Classificador Aleatório (0.5000)')
    axes[0, 0].fill_between(fpr, tpr, alpha=0.12, color='#1f4e78')
    axes[0, 0].set_title('Curva ROC e Área Sob a Curva (AUROC)', fontweight='bold')
    axes[0, 0].set_xlabel('Taxa de Falsos Positivos (1 - Especificidade)')
    axes[0, 0].set_ylabel('Taxa de Verdadeiros Positivos (Recall)')
    axes[0, 0].legend(loc='lower right')
    axes[0, 0].grid(True, linestyle=':', alpha=0.6)

    # 2. Teste Kolmogorov-Smirnov (KS2)
    thresholds_ks = np.linspace(0, 1, 200)
    prob_pos = y_prob[y_test == 1]
    prob_neg = y_prob[y_test == 0]
    cdf_pos = [np.mean(prob_pos <= t) for t in thresholds_ks]
    cdf_neg = [np.mean(prob_neg <= t) for t in thresholds_ks]
    diffs = np.abs(np.array(cdf_neg) - np.array(cdf_pos))
    idx_max = np.argmax(diffs)
    t_max = thresholds_ks[idx_max]

    axes[0, 1].plot(thresholds_ks, cdf_neg, label='CDF Classe Permanência (0)', color='#1f4e78', lw=2.2)
    axes[0, 1].plot(thresholds_ks, cdf_pos, label='CDF Classe Churn (1)', color='#e67e22', lw=2.2)
    axes[0, 1].vlines(t_max, cdf_pos[idx_max], cdf_neg[idx_max], color='#c0392b', linestyle='--', lw=2,
                      label=f'Separação Máxima KS2 = {res_base["metricas"]["ks2_stat"]:.4f} (limiar={t_max:.2f})')
    axes[0, 1].set_title(f'Teste Kolmogorov-Smirnov (KS2 = {res_base["metricas"]["ks2_stat"]:.4f} | p = {res_base["metricas"]["ks2_p_valor"]:.2e})', fontweight='bold')
    axes[0, 1].set_xlabel('Probabilidade Predita de Churn')
    axes[0, 1].set_ylabel('Proporção Acumulada (CDF)')
    axes[0, 1].legend(loc='upper left')
    axes[0, 1].grid(True, linestyle=':', alpha=0.6)

    # 3. Curva Precisão vs. Recall
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
    baseline_rate = y_test.mean()
    axes[1, 0].plot(rec_curve, prec_curve, color='#6ab04c', lw=2.5, label='Curva Precisão-Recall')
    axes[1, 0].axhline(baseline_rate, color='#c0392b', linestyle='--', lw=1.5, label=f'Baseline Aleatório ({baseline_rate*100:.1f}%)')
    axes[1, 0].set_title('Curva Precisão vs. Recall (Trade-off de Classificação)', fontweight='bold')
    axes[1, 0].set_xlabel('Recall (Revocação / Cobertura de Churn)')
    axes[1, 0].set_ylabel('Precisão (Assertividade dos Alertas)')
    axes[1, 0].legend(loc='lower left')
    axes[1, 0].grid(True, linestyle=':', alpha=0.6)

    # 4. Experimento max_depth vs. Overfitting
    depths_x = [d['hiperparametros']['max_depth'] for d in historico_depths]
    acc_tr_y = [d['metricas']['acuracia_treino'] for d in historico_depths]
    acc_te_y = [d['metricas']['acuracia'] for d in historico_depths]

    axes[1, 1].plot(depths_x, acc_tr_y, marker='o', color='#1f4e78', lw=2, label='Acurácia no Treino (Overfitting)')
    axes[1, 1].plot(depths_x, acc_te_y, marker='s', color='#e67e22', lw=2, label='Acurácia no Teste (Generalização)')
    axes[1, 1].axvline(5, color='#27ae60', linestyle='--', lw=1.8, label='Profundidade de Referência (max_depth = 5)')
    axes[1, 1].set_xticks(depths_x)
    axes[1, 1].set_title('Experimento de Hiperparâmetros: max_depth vs. Overfitting', fontweight='bold')
    axes[1, 1].set_xlabel('Profundidade Máxima da Árvore (max_depth)')
    axes[1, 1].set_ylabel('Acurácia')
    axes[1, 1].legend(loc='lower left')
    axes[1, 1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close(fig)