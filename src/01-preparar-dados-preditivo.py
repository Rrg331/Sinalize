#ETL para modelo PREDITIVO - prevê falhas FUTURAS
#Processa os arquivos na pasta RAW e gera os dados em formato de feature 
 
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from glob import glob

# Parâmetros
if len(sys.argv) > 1:
    periodos = [int(d) for d in sys.argv[1].split(',')]
else:
    periodos = [30,35,40,45,50, 55, 60,65,70,75,80,85,90]



falhas = pd.read_csv('../data/raw/falhas.csv', sep=';', encoding='utf-8-sig')
manutencao = pd.read_csv('../data/raw/manutencao.csv', sep=';', encoding='utf-8-sig')
transformadores = pd.read_csv('../data/raw/transformadores.csv', sep=';', encoding='utf-8-sig', decimal=',')


falhas['inicio'] = pd.to_datetime(falhas['inicio'])
falhas['fim'] = pd.to_datetime(falhas['fim'])
manutencao['inicio'] = pd.to_datetime(manutencao['inicio'])
manutencao['fim'] = pd.to_datetime(manutencao['fim'])
transformadores['data_entrada_operacao'] = pd.to_datetime(transformadores['data_entrada_operacao'])

#remover dados anteriores a data de corte ( 2023-01-01)
falhas = falhas[falhas['inicio'] >= '2023-01-01']
manutencao = manutencao[manutencao['inicio'] >= '2023-01-01']


equipamentos_com_manutencao = manutencao['id_equipamento'].unique()
equipamentos_com_falha = falhas['id_equipamento'].unique()
equipamentos_validos = np.intersect1d(equipamentos_com_manutencao, equipamentos_com_falha)
print(f"Equipamentos válidos (com manutenção e falha): {len(equipamentos_validos)}")

falhas = falhas[falhas['id_equipamento'].isin(equipamentos_validos)]
manutencao = manutencao[manutencao['id_equipamento'].isin(equipamentos_validos)]
transformadores = transformadores[transformadores['id_equipamento'].isin(equipamentos_validos)]


print("Carregando dados de utilização...")
utilizacao_dict = {}
arquivos_utilizacao = glob('../data/raw/utilizacao_transformadores/*_P_AMP.v.csv')

for arquivo in arquivos_utilizacao:
    print(f'processando arquivo: {arquivo}')
    id_equipamento_utilizacao = os.path.basename(arquivo).replace('_P_AMP.v.csv', '')
    df_util = pd.read_csv(arquivo, sep=';', header=None, names=['timestamp', 'valor'], dtype=str, low_memory=False)
    df_util['timestamp'] = pd.to_datetime(df_util['timestamp'], format='%d/%m/%Y %H:%M:%S')
    

    df_util['valor'] = pd.to_numeric(df_util['valor'].str.replace(',','.'))
    df_util = df_util.dropna()
    utilizacao_dict[id_equipamento_utilizacao] = df_util
  

print(f"Carregados {len(utilizacao_dict)} arquivos de utilização")

id_to_ido = transformadores.set_index('id_equipamento')['id_equipamento_utilizacao'].to_dict()

def calcular_features_utilizacao(id_equipamento_utilizacao, data_referencia, janela_dias, limite_potencia):

    if id_equipamento_utilizacao not in utilizacao_dict:
        return {}
    
    df_util = utilizacao_dict[id_equipamento_utilizacao]
    data_fim = data_referencia
    data_inicio = data_referencia - timedelta(days=janela_dias)
    
    dados = df_util[
        (df_util['timestamp'] >= data_inicio) &
        (df_util['timestamp'] < data_fim)
    ]
    
    if len(dados) == 0:
        return {}
    
    valores = pd.to_numeric(dados['valor'], errors='coerce').dropna()
    
    if len(valores) == 0:
        return {}
    
    if pd.notna(limite_potencia) and limite_potencia > 0:
        sobrecargas = (valores > limite_potencia).sum()
    else:
        sobrecargas = 0
    
    return {
        'utilizacao_media': valores.mean(),
        'utilizacao_maxima': valores.max(),
        'utilizacao_minima': valores.min(),
        'utilizacao_desvio': valores.std() if len(valores) > 1 else 0, #todo: remover isso  não presta
        'qtd_sobrecargas': sobrecargas,
        'dias_com_dados_util': len(dados)
    }

def criar_features_preditivo(equipamentos, janela_previsao):   
    features_list = []
    data_hoje = datetime.now()
    data_referencia = data_hoje - timedelta(days=janela_previsao)
    
    print(f"Data: {data_referencia.date()}")
    print(f"Janela: {data_referencia.date()} até {data_hoje.date()}")
    
    for equip_id in equipamentos:
        trans_data = transformadores[transformadores['id_equipamento'] == equip_id]
        if trans_data.empty:
            continue
        
        trans_data = trans_data.iloc[0]
        
        idade = (data_referencia - trans_data['data_entrada_operacao']).days
        if idade < 0:
            continue
        
        manut_equip = manutencao[
            (manutencao['id_equipamento'] == equip_id) &
            (manutencao['fim'] <= data_referencia)
        ]
        num_manutencoes = len(manut_equip)
        
        if num_manutencoes == 0:
            continue
        
        if num_manutencoes > 1:
            manut_sorted = manut_equip.sort_values('inicio')
            intervalos = manut_sorted['inicio'].diff().dt.days.dropna()
            intervalo_medio = intervalos.mean() if len(intervalos) > 0 else 0
        else:
            intervalo_medio = 0
        
        falhas_historicas = falhas[
            (falhas['id_equipamento'] == equip_id) &
            (falhas['inicio'] <= data_referencia)
        ]
        num_falhas = len(falhas_historicas)
        taxa_falhas = num_falhas / (idade / 365) if idade > 0 else 0
        minutos_falha = falhas_historicas['duracao'].sum()
        taxa_minutos_falha = minutos_falha / (idade / 365) if idade > 0 else 0
        
        ultima_manut = manut_equip['fim'].max()
        dias_desde_manut = (data_referencia - ultima_manut).days
        if dias_desde_manut < 0:
            dias_desde_manut = 0
        
        falhas_futuras = falhas[
            (falhas['id_equipamento'] == equip_id) &
            (falhas['inicio'] > data_referencia) &
            (falhas['inicio'] <= data_hoje)
        ]
        vai_falhar = 1 if len(falhas_futuras) > 0 else 0
        
        id_equipamento_utilizacao = id_to_ido.get(equip_id)
        limite_pot = trans_data['limite_potencia'] if pd.notna(trans_data['limite_potencia']) else 0
        features_util = calcular_features_utilizacao(id_equipamento_utilizacao, data_referencia, 30, limite_pot) if id_equipamento_utilizacao else {}
        
        features = {
            'id_equipamento': equip_id,
            'idade_dias': idade,
            'num_manutencoes': num_manutencoes,
            'intervalo_medio_manut': intervalo_medio,
            'num_falhas_historico': num_falhas,
            'taxa_falhas_ano': taxa_falhas,
            'minutos_falha_historico': minutos_falha,
            'taxa_minutos_falha_ano': taxa_minutos_falha,
            'dias_desde_ultima_manut': dias_desde_manut,
            'limite_potencia': limite_pot,
            'utilizacao_media': features_util.get('utilizacao_media', 0),
            'utilizacao_maxima': features_util.get('utilizacao_maxima', 0),
            'utilizacao_minima': features_util.get('utilizacao_minima', 0),
            'utilizacao_desvio': features_util.get('utilizacao_desvio', 0),
            'qtd_sobrecargas': features_util.get('qtd_sobrecargas', 0),
            'dias_com_dados_util': features_util.get('dias_com_dados_util', 0),
            'vai_falhar': vai_falhar
        }
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)

inicio = pd.Timestamp.now()
for periodo in periodos:
    print(f"\nProcessando previsão para {periodo}")
    df = criar_features_preditivo(equipamentos_validos, periodo)
    df.to_csv(f'../data/gold/features_preditivo_{periodo}d.csv', index=False)

print(f"fim: {pd.Timestamp.now() - inicio} ")
