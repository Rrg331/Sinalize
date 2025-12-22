import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import sys
import numpy as np

if len(sys.argv) > 1:
    periodos = [int(d) for d in sys.argv[1].split(',')]
else:
    periodos = [30,35,40,45,50, 55, 60,65,70,75,80,85,90]

feature_names = ['idade_dias', 'num_manutencoes', 'intervalo_medio_manut', 
                 'num_falhas_historico', 'taxa_falhas_ano', 'minutos_falha_historico', 'taxa_minutos_falha_ano',
                 'dias_desde_ultima_manut', 'limite_potencia',
                 'utilizacao_media', 'utilizacao_maxima', 'utilizacao_minima', 'qtd_sobrecargas']

resultados = []
resumo_dados = []
importancias_todas = []
inicio = pd.Timestamp.now()

for periodo in periodos:
    print(f"Treinando modelo PREDITIVO para {periodo} dias no futuro")

    print("Carregando dados de features...")    
    df = pd.read_csv(f'../data/gold/features_preditivo_{periodo}d.csv')
    
    df_original = len(df)
    df = df[df['dias_com_dados_util'] > 0]
    print(f"Equipamentos: {df_original} -> {len(df)} (removidos {df_original - len(df)} sem dados)")
    
    X = df[feature_names]
    y = df['vai_falhar']
    
    total_equipamentos = len(df)
    total_falhas = y.sum()
    perc_falhas = (total_falhas / total_equipamentos) * 100
    
    print(f"Distribuição: Falhas={total_falhas} ({perc_falhas:.1f}%), OK={total_equipamentos-total_falhas} ({(total_equipamentos-total_falhas)/total_equipamentos*100:.1f}%)")
    
    # Armazenar resumo dos dados
    resumo_dados.append({
        'Período (dias)': periodo,
        'Total Equipamentos': total_equipamentos,
        'Falhas': total_falhas,
        '% Falhas': perc_falhas
    })
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"Após SMOTE: {len(y_train_bal)} amostras (Falhas={y_train_bal.sum()}, OK={len(y_train_bal)-y_train_bal.sum()})")
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    #otimização com grid search
    grid = GridSearchCV(rf, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
    grid.fit(X_train_bal, y_train_bal)
    
    model = grid.best_estimator_
    print(f"Melhores params: {grid.best_params_}")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    

    importancias = pd.DataFrame({
        'Feature': feature_names,
        'Importancia': model.feature_importances_
    }).sort_values('Importancia', ascending=False)
    
    # Armazenar importâncias para CSV final
    for i, row in importancias.iterrows():
        importancias_todas.append({
            'Feature': row['Feature'],
            'Dias': periodo,
            'Importancia': row['Importancia']
        })
    
    print("\nImportância das Features:")
    print(importancias.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    print(f"\nMétricas TREINO - tem que dar 100%:")
    print(f"  ROC-AUC: {roc_auc_score(y_train, y_train_proba):.3f}")
    print(f"  Accuracy: {accuracy_score(y_train, y_train_pred):.3f}")
    
    print(f"\nMétricas TESTE - métrica real:")
    print(classification_report(y_test, y_pred, target_names=['OK', 'FALHA']))
    
    resultados.append({
        'Dias': periodo,
        'ROC-AUC': roc_auc_score(y_test, y_proba),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Falhas': y.sum(),
        '% Falhas': y.sum()/len(y)*100
    })
    
    joblib.dump(model, f'../models/rf/preditivo_{periodo}d.pkl')

print(f"\n{'='*90}")
print("RESUMO DOS DADOS POR PERÍODO")
print('='*90)
df_resumo = pd.DataFrame(resumo_dados)
print(df_resumo.to_string(index=False, float_format=lambda x: f'{x:.1f}' if x != int(x) else f'{int(x)}'))
df_resumo.to_csv('../data/gold/resumo_dados_periodos.csv', index=False)
print("\nResumo salvo em: ../data/gold/resumo_dados_periodos.csv")

print(f"\n{'='*90}")
print("RESULTADOS DO TREINAMENTO - MODELO PREDITIVO")

df_resultados = pd.DataFrame(resultados)
print(df_resultados.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

print(f"\n{'='*90}")
print("IMPORTÂNCIA DAS FEATURES POR PERÍODO")
print('='*90)
df_importancias = pd.DataFrame(importancias_todas)
print(df_importancias.to_string(index=False, float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else str(x)))
df_importancias.to_csv('../data/gold/importancias_features.csv', index=False)
print("\nImportâncias salvas em: ../data/gold/importancias_features.csv")
print('='*90)
print(f"Tempo total: {pd.Timestamp.now() - inicio}")
print("Modelos salvos em: ../models/rf/")

