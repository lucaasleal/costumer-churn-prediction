import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

# 1. Leitura da planilha pré-processada
filepath = './Entrega_1/base_churn_5000_VF2_aluno_preprocessada.xlsx'
xls = pd.ExcelFile(filepath)
df = pd.read_excel(filepath, sheet_name=xls.sheet_names[0])

# 2. Saneamento de outliers residuais em mensalidade_rs (< 10 ou > 500) e imputação KNN
df_corr = df.copy()
outliers_mensalidade = (df_corr['mensalidade_rs'] < 10.0) | (df_corr['mensalidade_rs'] > 500.0)
df_corr.loc[outliers_mensalidade, 'mensalidade_rs'] = np.nan

colunas_num = [
    'mensalidade_rs', 'atrasos_6m', 'fibra_optica', 'debito_automatico',
    'tempo_cliente_meses', 'numero_chamados_total', 'numero_chamados_30_dias',
    'numero_chamados_60_dias', 'numero_chamados_90_dias'
]
imputer = KNNImputer(n_neighbors=5, metric='nan_euclidean')
df_corr[colunas_num] = imputer.fit_transform(df_corr[colunas_num])

# 3. Matriz 1: Variáveis Originais + One-Hot Encoding vs. Churn
df_orig_encoded = pd.get_dummies(
    df_corr.drop(columns=['id_cliente']),
    columns=['tipo_contrato', 'forma_pagamento'],
    dtype=int
)
ordem_orig = df_orig_encoded.corr()['churn'].sort_values(ascending=False).index.tolist()
matriz_corr_orig = df_orig_encoded[ordem_orig].corr()

# 4. Criação dos Bins Preditivos (Variáveis Discretizadas) e Combo Risco Triplo
df_bins = pd.DataFrame(index=df_corr.index)

# Bins de Tempo de Casa (cortes da Árvore de Decisão)
df_bins['bin_novato_ate_12m'] = (df_corr['tempo_cliente_meses'] <= 12).astype(int)
df_bins['bin_tempo_ate_18m'] = (df_corr['tempo_cliente_meses'] <= 18).astype(int)
df_bins['bin_veterano_acima_24m'] = (df_corr['tempo_cliente_meses'] > 24).astype(int)

# Bins de Chamados ao Suporte
df_bins['bin_chamados_2ouMais'] = (df_corr['numero_chamados_total'] >= 2).astype(int)
df_bins['bin_chamados_critico_3ouMais'] = (df_corr['numero_chamados_total'] >= 3).astype(int)

# Bins de Atrasos Financeiros em 6 Meses
df_bins['bin_atrasos_1ouMais'] = (df_corr['atrasos_6m'] >= 1).astype(int)
df_bins['bin_atrasos_critico_2ouMais'] = (df_corr['atrasos_6m'] >= 2).astype(int)

# Bin de Mensalidade Elevada (>= R$ 85,00)
df_bins['bin_mensalidade_acima_85'] = (df_corr['mensalidade_rs'] >= 85.0).astype(int)

# Indicadores Binários de Contrato e Serviço
df_bins['bin_contrato_mensal'] = (df_corr['tipo_contrato'] == 'Mensal').astype(int)
df_bins['bin_contrato_bienal'] = (df_corr['tipo_contrato'] == 'Bienal').astype(int)
df_bins['bin_fibra_optica'] = (df_corr['fibra_optica'] == 1).astype(int)
df_bins['bin_sem_debito_auto'] = (df_corr['debito_automatico'] == 0).astype(int)

# Features de Interação Não-Linear: COMBO RISCO TRIPLO
df_bins['combo_risco_triplo_estrutural'] = (
    (df_bins['bin_contrato_mensal'] == 1) &
    (df_bins['bin_tempo_ate_18m'] == 1) &
    (df_bins['bin_chamados_2ouMais'] == 1)
).astype(int)

df_bins['combo_risco_triplo_operacional'] = (
    (df_bins['bin_contrato_mensal'] == 1) &
    (df_bins['bin_chamados_2ouMais'] == 1) &
    (df_bins['bin_atrasos_1ouMais'] == 1)
).astype(int)

soma_fatores_criticos = (
    df_bins['bin_contrato_mensal'] +
    df_bins['bin_tempo_ate_18m'] +
    df_bins['bin_chamados_2ouMais'] +
    df_bins['bin_atrasos_critico_2ouMais']
)
df_bins['combo_risco_triplo_acumulado'] = (soma_fatores_criticos >= 3).astype(int)
df_bins['churn'] = df_corr['churn'].astype(int)

ordem_bins = df_bins.corr()['churn'].sort_values(ascending=False).index.tolist()
matriz_corr_bins = df_bins[ordem_bins].corr()

# Diagnóstico de Taxa de Churn e Lift por Bin / Combo
resumo_bins = []
taxa_media = float(df_bins['churn'].mean() * 100)
for col in ordem_bins:
    if col == 'churn':
        continue
    mask = df_bins[col] == 1
    qtd = int(mask.sum())
    pct_base = float(qtd / len(df_bins) * 100)
    tx_churn_sim = float(df_bins.loc[mask, 'churn'].mean() * 100) if qtd > 0 else 0.0
    tx_churn_nao = float(df_bins.loc[~mask, 'churn'].mean() * 100) if (~mask).sum() > 0 else 0.0
    lift = float(tx_churn_sim / taxa_media)
    corr_val = float(matriz_corr_bins.loc[col, 'churn'])
    resumo_bins.append({
        'feature_bin': col,
        'correlacao_pearson_churn': round(corr_val, 4),
        'clientes_no_bin': qtd,
        'percentual_da_base': round(pct_base, 2),
        'taxa_churn_no_bin_pct': round(tx_churn_sim, 2),
        'taxa_churn_fora_do_bin_pct': round(tx_churn_nao, 2),
        'lift_risco': round(lift, 2)
    })

df_resumo_bins = pd.DataFrame(resumo_bins)

# 5. Geração e Salvamento dos Gráficos
os.makedirs('./Entrega_1/results/correlacao', exist_ok=True)
os.makedirs('./Entrega_1/results/relatorios', exist_ok=True)

# Figura 1: Heatmap Variáveis Originais
fig1, ax1 = plt.subplots(figsize=(13, 10), facecolor='white')
mask_tri1 = np.triu(np.ones_like(matriz_corr_orig, dtype=bool), k=1)
sns.heatmap(
    matriz_corr_orig, mask=mask_tri1, annot=True, fmt='.2f', cmap='RdBu_r',
    vmin=-0.5, vmax=0.5, center=0, square=True, linewidths=0.6,
    cbar_kws={'shrink': 0.8, 'label': 'Coeficiente de Correlação de Pearson (r)'},
    annot_kws={'size': 8.5}, ax=ax1
)
ax1.set_title('Matriz de Correlação de Pearson — Variáveis Originais e One-Hot vs. Churn (N=5004)',
              fontsize=13, fontweight='bold', pad=18, color='#1a252f')
plt.xticks(rotation=40, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('./Entrega_1/results/correlacao/matriz_correlacao_variaveis_originais.png', dpi=300, bbox_inches='tight')
plt.close(fig1)

# Figura 2: Heatmap Bins Preditivos e Combo Risco Triplo
cols_foco = [
    'churn', 'combo_risco_triplo_acumulado', 'combo_risco_triplo_operacional',
    'combo_risco_triplo_estrutural', 'bin_contrato_mensal', 'bin_tempo_ate_18m',
    'bin_novato_ate_12m', 'bin_chamados_critico_3ouMais', 'bin_chamados_2ouMais',
    'bin_atrasos_critico_2ouMais', 'bin_fibra_optica', 'bin_mensalidade_acima_85',
    'bin_sem_debito_auto', 'bin_contrato_bienal', 'bin_veterano_acima_24m'
]
matriz_foco = df_bins[cols_foco].corr()

fig2, ax2 = plt.subplots(figsize=(14, 11), facecolor='white')
mask_tri2 = np.triu(np.ones_like(matriz_foco, dtype=bool), k=1)
sns.heatmap(
    matriz_foco, mask=mask_tri2, annot=True, fmt='.2f', cmap='RdBu_r',
    vmin=-0.5, vmax=0.6, center=0, square=True, linewidths=0.6,
    cbar_kws={'shrink': 0.8, 'label': 'Coeficiente de Correlação de Pearson (r)'},
    annot_kws={'size': 8.5, 'weight': 'bold'}, ax=ax2
)
ax2.set_title('Matriz de Correlação de Pearson — Bins Preditivos e Combo Risco Triplo vs. Churn (N=5004)',
              fontsize=13, fontweight='bold', pad=18, color='#1a252f')
plt.xticks(rotation=40, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('./Entrega_1/results/correlacao/matriz_correlacao_bins_combo_triplo.png', dpi=300, bbox_inches='tight')
plt.close(fig2)

# Figura 3: Comparativo Lado a Lado (Correlação de Pearson vs Taxa de Churn %)
fig3, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 7.5), facecolor='white')
df_plot = df_resumo_bins.sort_values(by='correlacao_pearson_churn', ascending=True)
cores_corr = ['#c0392b' if 'combo' in f else ('#e67e22' if v > 0 else '#2980b9')
              for f, v in zip(df_plot['feature_bin'], df_plot['correlacao_pearson_churn'])]

bars_l = ax_left.barh(df_plot['feature_bin'], df_plot['correlacao_pearson_churn'], color=cores_corr, height=0.68)
ax_left.axvline(0, color='#2c3e50', lw=1.2)
ax_left.set_xlim(-0.32, 0.44)
ax_left.set_title('Correlação Linear de Pearson (r) com o Churn', fontsize=12, fontweight='bold', color='#1a252f')
ax_left.set_xlabel('Coeficiente de Pearson (r)', fontsize=10.5, fontweight='bold')
ax_left.grid(True, axis='x', linestyle=':', alpha=0.6)
for bar in bars_l:
    w = bar.get_width()
    offset = 0.01 if w >= 0 else -0.01
    ha_align = 'left' if w >= 0 else 'right'
    ax_left.text(w + offset, bar.get_y() + bar.get_height()/2, f'{w:+.3f}',
                 va='center', ha=ha_align, fontsize=9, fontweight='bold', color='#1a252f')

bars_r = ax_right.barh(df_plot['feature_bin'], df_plot['taxa_churn_no_bin_pct'], color=cores_corr, height=0.68)
ax_right.axvline(taxa_media, color='#2c3e50', linestyle='--', lw=1.8, label=f'Taxa Média da Base ({taxa_media:.2f}%)')
ax_right.set_xlim(0, 85)
ax_right.set_title('Taxa de Churn (%) Dentro de Cada Faixa / Combo', fontsize=12, fontweight='bold', color='#1a252f')
ax_right.set_xlabel('Taxa de Cancelamento / Churn (%)', fontsize=10.5, fontweight='bold')
ax_right.legend(loc='lower right')
ax_right.grid(True, axis='x', linestyle=':', alpha=0.6)
for bar in bars_r:
    w = bar.get_width()
    ax_right.text(w + 1.2, bar.get_y() + bar.get_height()/2, f'{w:.1f}%',
                  va='center', ha='left', fontsize=9, fontweight='bold', color='#1a252f')

plt.tight_layout()
plt.savefig('./Entrega_1/results/correlacao/comparativo_correlacao_e_taxa_churn.png', dpi=300, bbox_inches='tight')
plt.close(fig3)

# 6. Exportação dos Resultados em JSON
relatorio_corr = {
    "amostra_analisada": int(len(df_bins)),
    "taxa_churn_media_base_pct": round(taxa_media, 2),
    "correlacao_variaveis_originais_com_churn": matriz_corr_orig['churn'].round(4).to_dict(),
    "diagnostico_bins_e_combo_triplo": resumo_bins,
    "matriz_correlacao_bins_completa": matriz_foco.round(4).to_dict()
}
with open('./Entrega_1/results/correlacao_pearson_bins.json', 'w', encoding='utf-8') as f:
    json.dump(relatorio_corr, f, indent=4, ensure_ascii=False)

print("Análise de correlação de Pearson concluída com sucesso!")
