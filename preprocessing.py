import pandas as pd
import numpy as np
import unicodedata

df = pd.read_excel('base_churn_5000_VF2_aluno.xlsx')

with pd.option_context('display.max_columns', None):
    print(df.head())

# Granularidade do número de chamadas
df['numero_chamadas'] = df['data_chamado'].fillna('').str.split(',').str.len().astype(int)

#Assumimos que se for NaN, o cliente não atrasou em nenhum momento (ANALISAR POSSÍVEIS USO DE MÉDIA/MEDIANA/KNN)
df['atrasos_6m'] = df['atrasos_6m'].fillna(0.0).astype(int)

#Se houve ocorrência de churn, 1, senão, 0.
df['churn'] = df['data_ocorrencia_churn'].notna().astype(int)

df = df.drop(columns = "data_chamado")

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
    print(df.head())

df.to_excel('base_churn_5000_VF2_aluno_preprocessada.xlsx', index=False)

print(df.columns)