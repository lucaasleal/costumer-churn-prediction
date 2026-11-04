import pandas as pd
import numpy as np
import unicodedata

df = pd.read_excel('base_churn_5000_VF2_aluno.xlsx')

with pd.option_context('display.max_columns', None):
    print(df.head())

# definição das datas base e de corte (final)
data_corte = pd.to_datetime("26-11-2026", format="%d-%m-%Y")
data_base = data_corte - pd.Timedelta(days=60)

# transformação dos valores para datetime
df['data_aquisicao_cliente'] = pd.to_datetime(df['data_aquisicao_cliente'], format="mixed", dayfirst=True)
df['data_ocorrencia_churn'] = pd.to_datetime(df['data_ocorrencia_churn'], format="mixed", dayfirst=True)

# cria novo conjunto de clientes que deram churn ou não somente após a data base
df_60 = df[((df['data_ocorrencia_churn'] >= data_base) | df['data_ocorrencia_churn'].isna())]

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

# cria coluna com a contagem do número de chamados até a data limite
df_60['numero_chamados_total'] = df_60['lista_chamados'].apply(
    lambda datas: sum(d <= data_corte for d in datas)
    if isinstance(datas, pd.DatetimeIndex)
    else 0
)

# cria coluna com a contagem do número de chamados entre a data base e a data limite
df_60['numero_chamados_30_dias'] = df_60['lista_chamados'].apply(
    lambda datas: sum(data_corte - pd.Timedelta(days=30) <= d <= data_corte for d in datas)
    if isinstance(datas, pd.DatetimeIndex)
    else 0
)


# cria coluna com a contagem do número de chamados entre a data base e a data limite
df_60['numero_chamados_60_dias'] = df_60['lista_chamados'].apply(
    lambda datas: sum(data_base <= d <= data_corte for d in datas)
    if isinstance(datas, pd.DatetimeIndex)
    else 0
)

# cria coluna com a contagem do número de chamados entre a data base e a data limite
df_60['numero_chamados_90_dias'] = df_60['lista_chamados'].apply(
    lambda datas: sum(data_corte - pd.Timedelta(days=90) <= d <= data_corte for d in datas)
    if isinstance(datas, pd.DatetimeIndex)
    else 0
)

#Assumimos que se for NaN, o cliente não atrasou em nenhum momento (ANALISAR POSSÍVEIS USO DE MÉDIA/MEDIANA/KNN)
df_60['atrasos_6m'] = df_60['atrasos_6m'].fillna(0.0).astype(int)

#Se houve ocorrência de churn, 1, senão, 0.
df_60['churn'] = df_60['data_ocorrencia_churn'].notna().astype(int)

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
colunas_string = df.select_dtypes(include=['object', 'string']).columns

for coluna in colunas_string:
    df[coluna] = df[coluna].apply(padronizar_string, coluna=coluna)

with pd.option_context('display.max_columns', None):
    print(df_60.head())

df.to_excel('base_churn_5000_VF2_aluno_preprocessada.xlsx', index=False)
