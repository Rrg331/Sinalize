import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime, timedelta
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


#obtém os dados brutos 
#filtra os dados para manter apenas os equipamentos que possuem tanto falhas quanto manutenções registradas
##separado em um arquivo pra facilitar a proveniencia de dados
#returna uma QUINTUPLA??  com os datasets necessários para o processamento das features
#falhas, manutencao, 
def obterdadosraw():

    #region DADOS csv raw

    print('carregando arquivos raw [falhas.csv]...')
    falhas = pd.read_csv('../data/raw/falhas.csv', sep=';', header=None , names=['id_equipamento', 'inicio', 'fim', 'duracao', 'tipo_falha'], parse_dates=['inicio', 'fim'], dayfirst=False)
    print(f'falhas.csv carregado com sucesso  {len(falhas)} registros!')

    print('carregando arquivos raw [manutencao.csv]...')
    manutencao = pd.read_csv('../data/raw/manutencao.csv', sep=';', header=None , names=['id_equipamento', 'inicio', 'fim', 'duracao'], parse_dates=['inicio', 'fim'], dayfirst=False)
    print(f'manutencao.csv carregado com sucesso  {len(manutencao)} registros!')

    print('carregando arquivos raw [transformadores.csv]...')
    transformadores = pd.read_csv('../data/raw/transformadores.csv', sep=';', header=None , names=['id_equipamento', 'data_entrada_operacao', 'tipo_arranjo_subestacao', 'tensao_base_substacao'], parse_dates=['data_entrada_operacao'], dayfirst=False)
    print(f'transformadores.csv carregado com sucesso  {len(transformadores)} registros!')

    print('carregando arquivos raw [limites.csv]...')
    limites = pd.read_csv('../data/raw/limites.csv', sep=';', header=None , names=['id_equipamento', 'limite'] , dtype={'limite': np.float64} )
    print(f'limites.csv carregado com sucesso  {len(limites)} registros!')


    #endregion
      

    #region DADOS de UTILIZAÇÃO (carga)
    utilizacao = carregar_dados_utilizacao()
    #endregion

    return falhas, manutencao, transformadores, utilizacao, limites 

def criarfeatures(falhas, manutencao, transformadores, utilizacao , limites, janela_previsao):   

    
    #considerando somente equipamentos que tiveram pelo menos uma manutenção na janela de dados
    equipamentos = manutencao['id_equipamento'].unique()
    
    features_list = []

    data_maxima_dados = max(falhas['inicio'].max(), manutencao['inicio'].max())

    data_referencia = data_maxima_dados - timedelta(days=janela_previsao)

    print(f'Calculando features para previsão de falhas em {janela_previsao} dias a partir de {data_referencia.date()} até: {data_maxima_dados.date()} ')

    
    for equip_id in equipamentos:
        print('Features.... Processando equipamento:', equip_id  )

        #region dados transformador
        trafo = transformadores[transformadores['id_equipamento'] == equip_id]        
        trafo = trafo.iloc[0] 
        idade = (data_referencia - trafo['data_entrada_operacao']).days

        limite = limites[limites['id_equipamento'] == equip_id] 

        limite = limite.iloc[0] if not limite.empty else None
        limite_pot = limite['limite'] if limite is not None else 0

        

        #endregion
  
        #region features manutencao 
        manut_equip = manutencao[
            (manutencao['id_equipamento'] == equip_id) &
            (manutencao['inicio'] < data_referencia)
        ]

        num_manutencoes = len(manut_equip)

        
        if num_manutencoes > 1:
            manut_sorted = manut_equip.sort_values('inicio')
            intervalos = manut_sorted['inicio'].diff().dt.days.dropna()
            intervalo_medio = intervalos.mean() 
        else:
            intervalo_medio = 0


        
        ultima_manut = manut_equip['inicio'].max()

        if pd.isna(ultima_manut):
            dias_desde_manut = 0
        else:
            dias_desde_manut = (data_referencia - ultima_manut).days
            if dias_desde_manut < 0:
                dias_desde_manut = 0


        #endregion

        #region features  falhas 
        falhas_historicas = falhas[
            (falhas['id_equipamento'] == equip_id) &
            (falhas['inicio'] < data_referencia)
        ]

        num_falhas = len(falhas_historicas)
        minutos_falha = falhas_historicas['duracao'].sum()


        if idade > 0 :
            taxa_falhas = num_falhas / (idade / 365) 
            taxa_minutos_falha = minutos_falha / (idade / 365)

        else:
            taxa_falhas = 0
            taxa_minutos_falha = 0


        falhasjanela = falhas[
            (falhas['id_equipamento'] == equip_id) &
            (falhas['inicio'] >= data_referencia) &
            (falhas['inicio'] <= data_maxima_dados)
        ]

        vai_falhar = 1 if len(falhasjanela) > 0 else 0

        #endregion

        
        #region features utilização 
        
         
        dfutilizacao = utilizacao.get(equip_id)

        if dfutilizacao is None or dfutilizacao.empty:
            print('Aviso: Dados de utilização não encontrados para o equipamento:', equip_id    )
            continue
        

        dadosutilizacao = dfutilizacao[
            (dfutilizacao['timestamp'] < data_referencia)
        ]

        medidas = dadosutilizacao['valor']
         
        if limite_pot > 0:
            sobrecargas = (medidas > limite_pot).sum()
        else:
            sobrecargas = 0

        
        features_utilizacao =  {
            'utilizacao_media': medidas.mean() if len(medidas) > 0 else 0,
            'utilizacao_maxima': medidas.max() if len(medidas) > 0 else 0,
            'utilizacao_minima': medidas.min() if len(medidas) > 0 else 0,
            'utilizacao_desvio': medidas.std() if len(medidas) > 1 else 0,
            'qtd_sobrecargas': sobrecargas,
            'dias_com_dados_util': len(dadosutilizacao['timestamp'].dt.date.unique())
        } 
        
        #endregion
        
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
            'utilizacao_media': features_utilizacao.get('utilizacao_media', 0),
            'utilizacao_maxima': features_utilizacao.get('utilizacao_maxima', 0),
            'utilizacao_minima': features_utilizacao.get('utilizacao_minima', 0),
            'utilizacao_desvio': features_utilizacao.get('utilizacao_desvio', 0),
            'qtd_sobrecargas': features_utilizacao.get('qtd_sobrecargas', 0),
            'dias_com_dados_util': features_utilizacao.get('dias_com_dados_util', 0),
            'vai_falhar': vai_falhar
        }

         
        features_list.append(features)
        print(f'Features do equipamento {equip_id} calculadas com sucesso!' )
    
    return pd.DataFrame(features_list)

def carregar_dados_utilizacao():
    arquivos_utilizacao = glob('../data/raw/utilizacao_transformadores/*.csv')
    total = len(arquivos_utilizacao)
    
    print(f'Carregando {total} arquivos de utilização em paralelo...')
    
    utilizacao = {}
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(carregar_arquivo_utilizacao, arq): arq for arq in arquivos_utilizacao}
        
        for i, future in enumerate(as_completed(futures), 1):
            id_equip, df = future.result()
            utilizacao[id_equip] = df
            
            if i % 100 == 0:
                print(f'Processados {i}/{total} arquivos ({i/total*100:.1f}%)')
    
    print(f'Todos os {total} arquivos de utilização carregados com sucesso!')
    return utilizacao

def carregar_arquivo_utilizacao(arquivo):
    id_equipamento = os.path.basename(arquivo).replace('.csv', '')
    df_util = pd.read_csv(
        arquivo, sep=';', header=None, 
        names=['timestamp', 'valor'], 
        parse_dates=['timestamp'], 
        dayfirst=False, 
        dtype={'valor': np.float64}
    )
    return id_equipamento, df_util.dropna()


