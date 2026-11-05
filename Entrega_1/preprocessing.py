import pandas as pd
import numpy as np
import unicodedata

df = pd.read_excel('./Entrega_1/base_churn_5000_VF2_aluno.xlsx')

with pd.option_context('display.max_columns', None):
    print(df.head())

# definição das datas base e de corte (final)
data_base = pd.to_datetime("21-09-2026", format="%d-%m-%Y")
data_corte = data_base + pd.Timedelta(days=60)

# transformação dos valores para datetime
df['data_aquisicao_cliente'] = pd.to_datetime(df['data_aquisicao_cliente'], format="mixed", dayfirst=True)
df['data_ocorrencia_churn'] = pd.to_datetime(df['data_ocorrencia_churn'], format="mixed", dayfirst=True)

# cria novo conjunto de clientes que deram churn ou não somente após a data base
df_60 = df[
    ((df['data_ocorrencia_churn'] >= data_base) 
    | df['data_ocorrencia_churn'].isna())
].copy()

# define a data final como sendo ou a data do churn ou a data de corte (caso em que o cliente não deu churn)
df_60['data_final'] = df_60['data_ocorrencia_churn'].fillna(data_corte)

# cria uma coluna que informa os dias ativos de tal cliente
df_60['dias_ativo'] = (df_60['data_final'] - df_60['data_aquisicao_cliente']).dt.days

# remove os cliente com dias ativos menor que 60 dias
df_60 = df_60[df_60['dias_ativo'] >= 60].copy()

# cria uma coluna de meses ativos
df_60['tempo_cliente_meses'] = (df_60['dias_ativo'] / 30).astype(int)

# Granularidade do número de chamadas --------------------------------------------------------- #

# cria coluna com lista de datas de chamado, que é uma string em data_chamado
df_60['lista_chamados'] = df_60['data_chamado'].dropna().str.split(r',\s*').apply(lambda lista: pd.to_datetime(lista, format='%d/%m/%Y'))

# cria coluna para armazenar a quantidade total de chamados
df_60['numero_chamados_total'] = df_60.apply(
    lambda row: sum(
        row['data_aquisicao_cliente'] <= d <= data_base
        for d in row['lista_chamados']
    )
    if isinstance(row['lista_chamados'], pd.DatetimeIndex)
    else 0,
    axis=1,
)

# cria três colunas para armazenar o total de chamados nos períodos determinados
for dias in [30, 60, 90]:
    data_inicio = data_base - pd.Timedelta(days=dias)
    df_60[f'numero_chamados_{dias}_dias'] = df_60.apply(
        lambda row: sum(
            (row['data_aquisicao_cliente'] <= d <= data_base)
            and (d >= data_inicio)
            for d in row['lista_chamados']
        )
        if isinstance(row['lista_chamados'], pd.DatetimeIndex)
        else 0,
        axis=1,
    )

#Assumimos que se for NaN, o cliente não atrasou em nenhum momento (ANALISAR POSSÍVEIS USO DE MÉDIA/MEDIANA/KNN)
df_60['atrasos_6m'] = df_60['atrasos_6m'].replace(99, np.nan).fillna(0.0).astype(int)

#Se houve ocorrência de churn, 1, senão, 0.
df_60['churn'] = (df_60['data_ocorrencia_churn'].notna() & (df_60['data_ocorrencia_churn'] <= data_corte)).astype(int)

# deleta colunas que não serão utilizadas
df_60 = df_60.drop(columns = ['data_chamado', 'lista_chamados', 'data_aquisicao_cliente', 'data_ocorrencia_churn', 'data_final', 'dias_ativo'])

def padronizar_string(valor, coluna=None):
    if pd.isna(valor):
        return valor

    contratos = {
        'mensal': 'Mensal',
        'anual': 'Anual',
        'bienal': 'Bienal'
    }

    pagamentos = {
        'boleto': 'Boleto',
        'cartao': 'Cartão',
        'pix': 'Pix'
    }

    valor = str(valor).strip().lower()
    valor = unicodedata.normalize('NFKD', valor)
    valor = ''.join(
        caractere for caractere in valor
        if not unicodedata.combining(caractere)
    )

    if coluna == 'tipo_contrato':
        return contratos[valor]
    elif coluna == 'forma_pagamento':
        return pagamentos[valor]
    else:
        return valor


# Seleciona todas as colunas do tipo string
colunas_string = df_60.select_dtypes(include=['object', 'string']).columns

for coluna in colunas_string:
    df_60[coluna] = df_60[coluna].apply(padronizar_string, coluna=coluna)

with pd.option_context('display.max_columns', None):
    print(df_60.head())

df_60.to_excel('./Entrega_1/base_churn_5000_VF2_aluno_preprocessada.xlsx', index=False)

print(df.columns)
