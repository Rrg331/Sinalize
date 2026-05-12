from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Carregar modelos disponíveis dinamicamente — RF e XGBoost
TIPOS_MODELO = ['rf', 'xgb']
periodos_disponiveis = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]

modelos = {tipo: {} for tipo in TIPOS_MODELO}
for tipo in TIPOS_MODELO:
    for periodo in periodos_disponiveis:
        path = f'../models/{tipo}/preditivo_{periodo}d.pkl'
        if os.path.exists(path):
            modelos[tipo][periodo] = joblib.load(path)
    print(f"Modelos {tipo.upper()}: {len(modelos[tipo])} horizontes carregados")

if not modelos['rf']:
    raise FileNotFoundError("Nenhum modelo RF encontrado. Execute 02-treinar-modelo-preditivo.py primeiro.")

# Carregar thresholds otimizados para cada tipo
thresholds = {}
for tipo in TIPOS_MODELO:
    path = f'../data/gold/thresholds_{tipo}.csv'
    if os.path.exists(path):
        df_thr = pd.read_csv(path).set_index('periodo_dias')
        thresholds[tipo] = df_thr.to_dict(orient='index')
        print(f"Thresholds {tipo.upper()} carregados para {len(thresholds[tipo])} horizontes")
    else:
        thresholds[tipo] = {}
        print(f"Thresholds {tipo.upper()} não encontrados — usando 0.5 padrão.")

print(f"\nAPI pronta com {len(modelos)} modelos: {sorted(modelos.keys())}")

FEATURE_COLS = ['idade_dias', 'num_manutencoes', 'intervalo_medio_manut',
                'num_falhas_historico', 'taxa_falhas_ano', 'minutos_falha_historico',
                'taxa_minutos_falha_ano', 'dias_desde_ultima_falha', 'dias_desde_ultima_manut', 'limite_potencia',
                'utilizacao_media', 'utilizacao_maxima', 'utilizacao_minima', 'utilizacao_desvio', 'taxa_sobrecargas_ano',
                'p90_utilizacao', 'delta_utilizacao', 'utilizacao_tendencia_90d', 'dias_acima_80pct_limite']

@app.route('/prever_falha', methods=['POST'])
def prever_falha():
    data = request.json
    
    if not data:
        return jsonify({'error': 'Nenhum dado fornecido'}), 400
    
    required_features = ['idade_dias', 'num_manutencoes', 'intervalo_medio_manut',
                         'num_falhas_historico', 'taxa_falhas_ano', 'minutos_falha_historico',
                         'taxa_minutos_falha_ano', 'dias_desde_ultima_manut', 'limite_potencia',
                         'utilizacao_media', 'utilizacao_maxima', 'utilizacao_minima', 'utilizacao_desvio', 'taxa_sobrecargas_ano',
                         'p90_utilizacao', 'delta_utilizacao', 'utilizacao_tendencia_90d', 'dias_acima_80pct_limite']
    required = ['id_equipamento'] + required_features
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({'error': f'Campos ausentes: {missing}'}), 400
    
    try:
        X = pd.DataFrame([[data[col] for col in FEATURE_COLS]], columns=FEATURE_COLS)
    except Exception as e:
        return jsonify({'error': f'Erro ao processar features: {str(e)}'}), 400
    
    tipo_modelo = data.get('modelo', 'rf')
    if tipo_modelo not in TIPOS_MODELO or not modelos[tipo_modelo]:
        return jsonify({'error': f"Modelo '{tipo_modelo}' não disponível. Use: {TIPOS_MODELO}"}), 400

    periodos = data.get('periodos', sorted(modelos[tipo_modelo].keys()))
    estrategia = data.get('estrategia_threshold', 'f1_max')
    estrategia_col = {
        'padrao': 'threshold_padrao',
        'f1_max': 'threshold_f1_max',
        'recall_minimo': 'threshold_recall_minimo'
    }.get(estrategia, 'threshold_f1_max')

    previsoes = []

    for periodo in periodos:
        if periodo not in modelos[tipo_modelo]:
            continue

        modelo = modelos[tipo_modelo][periodo]
        proba = modelo.predict_proba(X)[0][1]
        thr = thresholds[tipo_modelo].get(periodo, {}).get(estrategia_col, 0.5)
        predicao = int(proba >= thr)

        # Incerteza: RF usa árvores individuais; XGBoost não tem estimators_
        if tipo_modelo == 'rf':
            tree_preds = np.array([t.predict_proba(X)[0][1] for t in modelo.estimators_])
            std = float(tree_preds.std())
        else:
            std = 0.0

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
        'modelos_disponiveis': TIPOS_MODELO,
        'periodos_disponiveis': sorted(modelos['rf'].keys()),
        'features_requeridas': FEATURE_COLS,
        'estrategias_threshold': {
            'padrao': 'threshold fixo 0.5',
            'f1_max': 'maximiza F1-Score (padrão) — recall médio ~44%',
            'recall_minimo': 'recall mínimo 50% — maior sensibilidade'
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
