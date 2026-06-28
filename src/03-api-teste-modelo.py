from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
from util.config import PERIODOS, FEATURE_NAMES

app = Flask(__name__)

# Modelos base carregados de disco; 'ensemble' é calculado em tempo de execução (RF + XGBoost)
TIPOS_MODELO = ['rf', 'xgb', 'rf_cal', 'ensemble']
MODELOS_DISCO = ['rf', 'xgb', 'rf_cal']
periodos_disponiveis = PERIODOS

modelos = {tipo: {} for tipo in TIPOS_MODELO}
for tipo in MODELOS_DISCO:
    for periodo in periodos_disponiveis:
        path = f'../models/{tipo}/preditivo_{periodo}d.pkl'
        if os.path.exists(path):
            modelos[tipo][periodo] = joblib.load(path)
    print(f"Modelos {tipo.upper()}: {len(modelos[tipo])} horizontes carregados")

if not modelos['rf']:
    raise FileNotFoundError("Nenhum modelo RF encontrado. Execute 02-treinar-modelo-preditivo.py primeiro.")

# Arquivos de threshold por tipo: ensemble usa thresholds.csv (médias RF+XGBoost)
THRESHOLD_FILES = {
    'rf':       '../data/gold/thresholds_rf.csv',
    'xgb':      '../data/gold/thresholds_xgb.csv',
    'rf_cal':   '../data/gold/thresholds_rf_cal.csv',
    'ensemble': '../data/gold/thresholds.csv',
}

thresholds = {}
for tipo in TIPOS_MODELO:
    path = THRESHOLD_FILES.get(tipo, '')
    if path and os.path.exists(path):
        df_thr = pd.read_csv(path).set_index('periodo_dias')
        thresholds[tipo] = df_thr.to_dict(orient='index')
        print(f"Thresholds {tipo.upper()} carregados para {len(thresholds[tipo])} horizontes")
    else:
        thresholds[tipo] = {}
        print(f"Thresholds {tipo.upper()} não encontrados — usando 0.5 padrão.")

print(f"\nAPI pronta com {len(modelos)} modelos: {sorted(modelos.keys())}")

FEATURE_COLS = FEATURE_NAMES

@app.route('/prever_falha', methods=['POST'])
def prever_falha():
    data = request.json
    
    if not data:
        return jsonify({'error': 'Nenhum dado fornecido'}), 400
    
    required_features = FEATURE_COLS
    required = ['id_equipamento'] + required_features
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({'error': f'Campos ausentes: {missing}'}), 400
    
    try:
        X = pd.DataFrame([[data[col] for col in FEATURE_COLS]], columns=FEATURE_COLS)
    except Exception as e:
        return jsonify({'error': f'Erro ao processar features: {str(e)}'}), 400
    
    tipo_modelo = data.get('modelo', 'rf')
    if tipo_modelo not in TIPOS_MODELO:
        return jsonify({'error': f"Modelo '{tipo_modelo}' não disponível. Use: {TIPOS_MODELO}"}), 400
    if tipo_modelo not in ('ensemble',) and not modelos[tipo_modelo]:
        return jsonify({'error': f"Modelo '{tipo_modelo}' sem horizontes carregados."}), 400

    # Para ensemble, usa os períodos que têm AMBOS rf e xgb disponíveis
    if tipo_modelo == 'ensemble':
        periodos_ensemble = sorted(set(modelos['rf'].keys()) & set(modelos['xgb'].keys()))
        periodos = data.get('periodos', periodos_ensemble)
    else:
        periodos = data.get('periodos', sorted(modelos[tipo_modelo].keys()))

    estrategia = data.get('estrategia_threshold', 'f1_max')
    estrategia_col = {
        'padrao': 'threshold_padrao',
        'f1_max': 'threshold_f1_max',
        'recall_minimo': 'threshold_recall_minimo'
    }.get(estrategia, 'threshold_f1_max')

    previsoes = []

    for periodo in periodos:
        if tipo_modelo == 'ensemble':
            if periodo not in modelos['rf'] or periodo not in modelos['xgb']:
                continue
            proba_rf  = modelos['rf'][periodo].predict_proba(X)[0][1]
            proba_xgb = modelos['xgb'][periodo].predict_proba(X)[0][1]
            proba = (proba_rf + proba_xgb) / 2.0
            # Incerteza: std entre as probabilidades dos dois modelos base
            std = float(np.std([proba_rf, proba_xgb]))
            thr = thresholds['ensemble'].get(periodo, {}).get(estrategia_col, 0.5)
        else:
            if periodo not in modelos[tipo_modelo]:
                continue
            modelo = modelos[tipo_modelo][periodo]
            proba = modelo.predict_proba(X)[0][1]
            thr = thresholds[tipo_modelo].get(periodo, {}).get(estrategia_col, 0.5)
            # RF expõe estimators_ individuais; rf_cal (CalibratedClassifierCV) não
            if tipo_modelo == 'rf' and hasattr(modelo, 'estimators_'):
                tree_preds = np.array([t.predict_proba(X)[0][1] for t in modelo.estimators_])
                std = float(tree_preds.std())
            else:
                std = 0.0

        predicao = int(proba >= thr)
        previsoes.append({
            'periodo_dias': periodo,
            'modelo': tipo_modelo,
            'probabilidade': round(float(proba), 4),
            'vai_falhar': predicao,
            'threshold_usado': round(float(thr), 4),
            'estrategia_threshold': estrategia,
            'desvio_padrao': round(std, 4),
            'intervalo_confianca': [round(float(proba - 1.96*std), 4), round(float(proba + 1.96*std), 4)]
        })
    
    return jsonify({
        'id_equipamento': data['id_equipamento'],
        'previsoes': previsoes
    })



@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'modelos': {tipo: sorted(m.keys()) for tipo, m in modelos.items()}
    })

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        'modelos_disponiveis': {
            'rf':      'Random Forest (base)',
            'xgb':     'XGBoost',
            'rf_cal':  'Random Forest calibrado (CalibratedClassifierCV — probabilidades mais confiáveis)',
            'ensemble': 'Média RF + XGBoost (melhor AUC médio: 0.671)'
        },
        'periodos_disponiveis': sorted(modelos['rf'].keys()),
        'features_requeridas': FEATURE_COLS,
        'estrategias_threshold': {
            'padrao': 'threshold fixo 0.5',
            'f1_max': 'maximiza F1-Score (padrão) — recall médio ~44%',
            'recall_minimo': 'recall minimo 50% — maior sensibilidade'
        },
        'exemplo_request': {
            'id_equipamento': 'TR-001',
            'idade_dias': 3650,
            'num_manutencoes': 5,
            'intervalo_medio_manut': 730,
            'num_falhas_historico': 2,
            'taxa_falhas_ano': 0.5,
            'minutos_falha_historico': 120,
            'taxa_minutos_falha_ano': 30,
            'dias_desde_ultima_falha': 365,
            'dias_desde_ultima_manut': 180,
            'limite_potencia': 100,
            'utilizacao_media': 75.5,
            'utilizacao_maxima': 95.0,
            'utilizacao_minima': 50.0,
            'utilizacao_desvio': 12.3,
            'taxa_sobrecargas_ano': 3,
            'p90_utilizacao': 88.5,
            'delta_utilizacao': 5.2,
            'utilizacao_tendencia_90d': 0.15,
            'dias_acima_80pct_limite': 45,
            'periodos': [30, 60, 90],
            'modelo': 'rf',
            'estrategia_threshold': 'f1_max'
        }
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("SINALIZE - API de Previsão de Falhas")
    print("="*60)
    print(f"Servidor: http://localhost:5001")
    print(f"Endpoints:")
    print(f"  POST /prever_falha - Realizar previsão")
    print(f"  GET  /health       - Status da API")
    print(f"  GET  /info         - Informações e exemplo")
    print("="*60 + "\n")
    app.run(debug=True, port=5001)
