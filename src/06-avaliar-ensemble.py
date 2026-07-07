import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from util.config import PERIODOS, FEATURE_NAMES

periodos = PERIODOS
feature_names = FEATURE_NAMES

resultados = []

for periodo in periodos:
    rf_path  = f'../models/rf/preditivo_{periodo}d.pkl'
    xgb_path = f'../models/xgb/preditivo_{periodo}d.pkl'
    csv_path = f'../data/gold/features_preditivo_{periodo}d.csv'

    try:
        rf  = joblib.load(rf_path)
        xgb = joblib.load(xgb_path)
        df  = pd.read_csv(csv_path)
    except FileNotFoundError as e:
        print(f"[{periodo}d] Arquivo não encontrado: {e}")
        continue

    df = df[df['dias_com_dados_util'] > 0]
    X = df[feature_names]
    y = df['vai_falhar']

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    proba_rf  = rf.predict_proba(X_test)[:, 1]
    proba_xgb = xgb.predict_proba(X_test)[:, 1]
    proba_ens = (proba_rf + proba_xgb) / 2.0

    auc_rf  = roc_auc_score(y_test, proba_rf)
    auc_xgb = roc_auc_score(y_test, proba_xgb)
    auc_ens = roc_auc_score(y_test, proba_ens)

    resultados.append({
        'Dias': periodo,
        'RF': round(auc_rf, 4),
        'XGBoost': round(auc_xgb, 4),
        'Ensemble': round(auc_ens, 4),
        'Δ_ens_vs_rf': round(auc_ens - auc_rf, 4),
    })








#saida 
df_res = pd.DataFrame(resultados)
print(f"\n{'='*65}")
print("ENSEMBLE RF + XGBoost — ROC-AUC por horizonte")
print('='*65)
print(df_res.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print(f"\nMédias:")
print(f"  RF:       {df_res['RF'].mean():.4f}")
print(f"  XGBoost:  {df_res['XGBoost'].mean():.4f}")
print(f"  Ensemble: {df_res['Ensemble'].mean():.4f}  (Δ = {df_res['Ensemble'].mean() - df_res['RF'].mean():+.4f} vs RF)")
print('='*65)

df_res.to_csv('../data/gold/metricas_ensemble.csv', index=False)
print("Métricas salvas em: ../data/gold/metricas_ensemble.csv")
