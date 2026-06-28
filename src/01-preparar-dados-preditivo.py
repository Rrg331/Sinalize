#ETL para modelo PREDITIVO - prevê falhas FUTURAS
#Processa os arquivos na pasta RAW e gera os dados em formato de feature  
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from util.funcoes import obterdadosraw, criarfeatures
from util.config import PERIODOS





print('carregando dataset raw ...')
falhas, manutencao, transformadores, dicutilizacao, limites  = obterdadosraw()
print('dados brutos carregados!')



# Parâmetros
periodos = [int(d) for d in sys.argv[1].split(',')] if len(sys.argv) > 1 else PERIODOS



inicio = pd.Timestamp.now()
for periodo in periodos:

    print(f"\nProcessando previsão para {periodo}")      
    df = criarfeatures(falhas, manutencao, transformadores, dicutilizacao , limites, periodo)
    df.to_csv(f'../data/gold/features_preditivo_{periodo}d.csv', index=False)

print(f"fim: {pd.Timestamp.now() - inicio} ")

