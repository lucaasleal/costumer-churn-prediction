import pandas as pd
import numpy as np

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

with pd.option_context('display.max_columns', None):
    print(df.head())