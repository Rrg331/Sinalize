#ETL para modelo PREDITIVO - prevê falhas FUTURAS
#Processa os arquivos na pasta RAW e gera os dados em formato de feature  
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from util.funcoes import obterdadosraw, criarfeatures





print('carregando dataset raw ...')
falhas, manutencao, transformadores, dicutilizacao, limites  = obterdadosraw()
print('dados brutos carregados!')



# Parâmetros
if len(sys.argv) > 1:
    periodos = [int(d) for d in sys.argv[1].split(',')]
else:
    periodos = [30,35,40,45,50, 55, 60,65,70,75,80,85,90]



inicio = pd.Timestamp.now()
for periodo in periodos:

    print(f"\nProcessando previsão para {periodo}")      
    df = criarfeatures(falhas, manutencao, transformadores, dicutilizacao , limites, periodo)
    df.to_csv(f'../data/gold/features_preditivo_{periodo}d.csv', index=False)

print(f"fim: {pd.Timestamp.now() - inicio} ")

