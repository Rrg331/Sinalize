from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

modelos = {
    30: joblib.load('../models/rf/preditivo_30d.pkl'),
    55: joblib.load('../models/rf/preditivo_55d.pkl'),
    90: joblib.load('../models/rf/preditivo_90d.pkl'),
    120: joblib.load('../models/rf/preditivo_120d.pkl')
}

FEATURE_COLS = ['idade_dias', 'num_manutencoes', 'intervalo_medio_manut',
                'num_falhas_historico', 'taxa_falhas_ano', 'minutos_falha_historico', 
                'taxa_minutos_falha_ano', 'dias_desde_ultima_manut', 'limite_potencia',
                'utilizacao_media', 'utilizacao_maxima', 'utilizacao_minima', 'qtd_sobrecargas']

@app.route('/prever_falha', methods=['POST'])
def prever_falha():
    data = request.json
    
    if not data:
        return jsonify({'error': 'Nenhum dado fornecido'}), 400
    
    required_features = ['idade_dias', 'num_manutencoes', 'intervalo_medio_manut',
                         'num_falhas_historico', 'taxa_falhas_ano', 'minutos_falha_historico', 
                         'taxa_minutos_falha_ano', 'dias_desde_ultima_manut', 'limite_potencia',
                         'utilizacao_media', 'utilizacao_maxima', 'utilizacao_minima', 'qtd_sobrecargas']
    required = ['id_equipamento'] + required_features
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({'error': f'Campos ausentes: {missing}'}), 400
    
    try:
        X = pd.DataFrame([[data[col] for col in FEATURE_COLS]], columns=FEATURE_COLS)
    except Exception as e:
        return jsonify({'error': f'Erro ao processar features: {str(e)}'}), 400
    
    periodos = data.get('periodos', [30, 55, 90, 120])
    previsoes = []
    
    for periodo in periodos:
        if periodo not in modelos:
            continue
            
        modelo = modelos[periodo]
        proba = modelo.predict_proba(X)[0][1]
        predicao = modelo.predict(X)[0]
        
        tree_predictions = np.array([tree.predict_proba(X)[0][1] for tree in modelo.estimators_])
        std = tree_predictions.std()
        
        previsoes.append({
            'periodo_dias': periodo,
            'probabilidade': round(float(proba), 4),
            'vai_falhar': int(predicao),
            'desvio_padrao': round(float(std), 4),
            'intervalo_confianca': [round(float(proba - 1.96*std), 4), round(float(proba + 1.96*std), 4)]
        })
    
    return jsonify({
        'id_equipamento': data['id_equipamento'],
        'previsoes': previsoes
    })



if __name__ == '__main__':
    app.run(debug=True, port=5001)
