import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Leitura automática da planilha bruta
filepath = 'base_churn_5000_VF2_aluno.xlsx'  # Altere o nome se necessário
xls = pd.ExcelFile(filepath)
df = pd.read_excel(filepath, sheet_name=xls.sheet_names[0])

N_total = len(df)

# ==============================================================================
# 2. CÁLCULO AUTOMÁTICO DAS 4 DIMENSÕES DE QUALIDADE
# ==============================================================================

# --- A) COMPLETUDE (% Preenchido sem NA nos atributos obrigatórios) ---
# Nota: data_chamado = NaN (0 chamados) e data_ocorrencia_churn = NaN (cliente ativo) são esperados
colunas_preenchimento = [
    'id_cliente',
    'data_aquisicao_cliente',
    'mensalidade_rs',
    'atrasos_6m',
    'tipo_contrato',
    'forma_pagamento',
    'fibra_optica',
    'debito_automatico',
]
mask_falha_completude = df[colunas_preenchimento].isna().any(axis=1)
qtd_falha_completude = mask_falha_completude.sum()
taxa_completude = ((N_total - qtd_falha_completude) / N_total) * 100

# --- B) VALIDADE (Aderência ao tipo, faixa numérica aceita e domínio canônico) ---
# Regras do Dicionário (Slide 4):
# - mensalidade_rs: [10.00, 500.00]
# - atrasos_6m: inteiro entre [0, 6]
# - tipo_contrato: {'Mensal', 'Anual', 'Bienal'}
# - forma_pagamento: {'Boleto', 'Cartão', 'Pix'}
# - fibra_optica / debito_automatico: {0, 1}
# - data_aquisicao_cliente: formato canônico DD/MM/YYYY (sem hífen '-')
inval_mensalidade = df['mensalidade_rs'].notna() & (
    (df['mensalidade_rs'] < 10.0) | (df['mensalidade_rs'] > 500.0)
)
inval_atrasos = df['atrasos_6m'].notna() & (
    (df['atrasos_6m'] < 0)
    | (df['atrasos_6m'] > 6)
    | (df['atrasos_6m'] % 1 != 0)
)
inval_contrato = df['tipo_contrato'].notna() & (
    ~df['tipo_contrato'].isin(['Mensal', 'Anual', 'Bienal'])
)
inval_pagamento = df['forma_pagamento'].notna() & (
    ~df['forma_pagamento'].isin(['Boleto', 'Cartão', 'Pix'])
)
inval_binarias = (~df['fibra_optica'].isin([0, 1])) | (
    ~df['debito_automatico'].isin([0, 1])
)
inval_formato_data = (
    df['data_aquisicao_cliente'].astype(str).str.contains('-', na=False)
)

mask_falha_validade = (
    inval_mensalidade
    | inval_atrasos
    | inval_contrato
    | inval_pagamento
    | inval_binarias
    | inval_formato_data
)
qtd_falha_validade = mask_falha_validade.sum()
taxa_validade = ((N_total - qtd_falha_validade) / N_total) * 100

# --- C) CONSISTÊNCIA (Regras cruzadas e coerência lógica/temporal) ---
aq_dt = pd.to_datetime(
    df['data_aquisicao_cliente'], format='mixed', dayfirst=True, errors='coerce'
)
ch_dt = pd.to_datetime(
    df['data_ocorrencia_churn'], format='mixed', dayfirst=True, errors='coerce'
)

# Regra 1: Churn não pode ocorrer antes da aquisição do cliente
incons_churn_aq = ch_dt.notna() & (ch_dt < aq_dt)

# Regra 2: Chamados ao suporte não podem ocorrer antes da aquisição do cliente
chamados_explodidos = df['data_chamado'].dropna().str.split(r',\s*').explode()
chamados_dt = pd.to_datetime(
    chamados_explodidos, format='%d/%m/%Y', errors='coerce'
)

df_temp_chamados = pd.DataFrame({
    'chamado_dt': chamados_dt,
    'aq_dt': aq_dt.loc[chamados_dt.index],
})
idx_incons_chamado = df_temp_chamados[
    df_temp_chamados['chamado_dt'] < df_temp_chamados['aq_dt']
].index.unique()
incons_chamado_aq = df.index.isin(idx_incons_chamado)

mask_falha_consistencia = incons_churn_aq | incons_chamado_aq
qtd_falha_consistencia = mask_falha_consistencia.sum()
taxa_consistencia = ((N_total - qtd_falha_consistencia) / N_total) * 100

# --- D) UNICIDADE (Ausência de duplicatas na chave primária) ---
mask_falha_unicidade = df.duplicated(subset=['id_cliente'], keep='first')
qtd_falha_unicidade = mask_falha_unicidade.sum()
taxa_unicidade = ((N_total - qtd_falha_unicidade) / N_total) * 100

# ==============================================================================
# 3. RELATÓRIO DE AUDITORIA NO TERMINAL
# ==============================================================================
print(f'=== RELATÓRIO DE AUDITORIA DA BASE (N = {N_total}) ===')
print(
    f'1. Completude:   {taxa_completude:.2f}% ({qtd_falha_completude} linhas'
    ' com NAs)'
)
print(
    '   - Detalhe NAs:'
    f" mensalidade_rs={df['mensalidade_rs'].isna().sum()},"
    f" atrasos_6m={df['atrasos_6m'].isna().sum()},"
    f" tipo_contrato={df['tipo_contrato'].isna().sum()},"
    f" forma_pagamento={df['forma_pagamento'].isna().sum()}"
)
print(
    f'2. Validade:     {taxa_validade:.2f}% ({qtd_falha_validade} linhas fora'
    ' de faixa/domínio)'
)
print(
    f'   - Detalhe Validade: mensalidade fora de [10, 500]={inval_mensalidade.sum()},'
    f' atrasos fora de [0, 6]={inval_atrasos.sum()},'
    f' contrato não-canônico={inval_contrato.sum()},'
    f' pagamento não-canônico={inval_pagamento.sum()},'
    f' data com hífen={inval_formato_data.sum()}'
)
print(
    f'3. Consistência: {taxa_consistencia:.2f}% ({qtd_falha_consistencia}'
    ' linhas com incoerência temporal)'
)
print(
    f'   - Detalhe Consistência: churn antes da aquisição={incons_churn_aq.sum()},'
    f' chamado antes da aquisição={incons_chamado_aq.sum()}'
)
print(
    f'4. Unicidade:    {taxa_unicidade:.2f}% ({qtd_falha_unicidade} linhas'
    ' duplicadas)'
)

# ==============================================================================
# 4. GERAÇÃO AUTOMÁTICA DO GRÁFICO
# ==============================================================================
categorias = [
    'Completude\n(% Preenchido)',
    'Validade\n(Faixa/Domínio)',
    'Consistência\n(Regras Cruzadas)',
    'Unicidade\n(Chave Única)',
]
valores = [taxa_completude, taxa_validade, taxa_consistencia, taxa_unicidade]

# Define Laranja (#d37012) para taxas < 99% e Teal (#129285) para taxas >= 99%
cores = ['#d37012' if v < 99.0 else '#129285' for v in valores]
cor_texto = '#1a252f'

fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
barras = ax.bar(categorias, valores, color=cores, width=0.55, zorder=3)

# Linha tracejada da Meta (100%) e indicação visual sem sobrepor as porcentagens
ax.axhline(y=100, color='#2c3e50', linestyle='--', linewidth=1.5, zorder=2)
ax.text(
    x=3.45,
    y=109,
    s='Meta de Conformidade (100%)',
    fontsize=10.5,
    fontweight='bold',
    color='#2c3e50',
    ha='right',
)
ax.annotate(
    '',
    xy=(3.3, 100.3),
    xytext=(3.3, 108),
    arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.2),
)

# Rótulos de dados no topo de cada barra
for barra in barras:
  altura = barra.get_height()
  pos_y = max(altura + 1.5, 101.5) if altura >= 95 else altura + 1.5
  ax.text(
      barra.get_x() + barra.get_width() / 2.0,
      pos_y,
      f'{altura:.2f}%',
      ha='center',
      va='bottom',
      fontsize=12,
      fontweight='bold',
      color=cor_texto,
  )

ax.set_ylim(0, 118)
ax.set_title(
    f'Diagnóstico das 4 Dimensões de Qualidade na Base Real (N={N_total})',
    fontsize=14,
    fontweight='bold',
    color=cor_texto,
    pad=25,
)
ax.set_ylabel(
    'Taxa de Conformidade (%)',
    fontsize=13,
    fontweight='bold',
    color=cor_texto,
    labelpad=15,
)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#7f8c8d')
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_color('#7f8c8d')
ax.spines['bottom'].set_linewidth(1.5)

ax.yaxis.grid(True, linestyle=':', color='#bdc3c7', alpha=0.8, zorder=1)
ax.tick_params(axis='x', labelsize=10, colors='#2c3e50')
ax.tick_params(axis='y', labelsize=11, colors='#2c3e50')

plt.tight_layout()
os.makedirs('results/auditoria', exist_ok=True)
plt.savefig(
    'results/auditoria/diagnostico_4_dimensoes.png',
    dpi=300,
    bbox_inches='tight',
)
plt.show()