import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.calibration import CalibratedClassifierCV
from util.config import PERIODOS, FEATURE_NAMES
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss

args = [a for a in sys.argv[1:] if a]
periodos_arg = next((a for a in args if a[0].isdigit()), None)
periodos = [int(d) for d in periodos_arg.split(',')] if periodos_arg else PERIODOS
feature_names = FEATURE_NAMES

OUTPUT_DIR = '../models/rf_cal'
os.makedirs(OUTPUT_DIR, exist_ok=True)

resultados = []

print(f"{'='*75}")
print(f"{'Dias':>4}  {'AUC orig':>9}  {'AUC cal':>8}  {'Brier orig':>11}  {'Brier cal':>10}  {'Δ Brier':>8}")
print(f"{'='*75}")

for periodo in periodos:
    rf_path  = f'../models/rf/preditivo_{periodo}d.pkl'
    csv_path = f'../data/gold/features_preditivo_{periodo}d.csv'

    try:
        model = joblib.load(rf_path)
        df    = pd.read_csv(csv_path)
    except FileNotFoundError as e:
        print(f"[{periodo}d] Arquivo não encontrado: {e}")
        continue

    df = df[df['dias_com_dados_util'] > 0]
    X  = df[feature_names]
    y  = df['vai_falhar']

    # Reproduz exatamente o split do treino
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Calibração: Platt scaling (sigmoid) ajustada no conjunto de treino original
    # cv='prefit' — modelo já treinado, apenas aprende o mapeamento sigmoidal
    cal_model = CalibratedClassifierCV(model, cv='prefit', method='sigmoid')
    cal_model.fit(X_train, y_train)

    proba_orig = model.predict_proba(X_test)[:, 1]
    proba_cal  = cal_model.predict_proba(X_test)[:, 1]

    auc_orig   = roc_auc_score(y_test, proba_orig)
    auc_cal    = roc_auc_score(y_test, proba_cal)
    brier_orig = brier_score_loss(y_test, proba_orig)
    brier_cal  = brier_score_loss(y_test, proba_cal)

    print(f"{periodo:>4}  {auc_orig:>9.4f}  {auc_cal:>8.4f}  "
          f"{brier_orig:>11.4f}  {brier_cal:>10.4f}  {brier_cal - brier_orig:>+8.4f}")

    joblib.dump(cal_model, f'{OUTPUT_DIR}/preditivo_{periodo}d.pkl')

    resultados.append({
        'Dias': periodo,
        'AUC_orig': round(auc_orig, 4),
        'AUC_cal': round(auc_cal, 4),
        'Brier_orig': round(brier_orig, 4),
        'Brier_cal': round(brier_cal, 4),
        #'delta_brier': round(brier_cal - round( brier_orig,2), 4),
        'delta_brier': round(brier_cal - brier_orig, 4),
    })

print(f"{'='*75}")
df_res = pd.DataFrame(resultados)
print(f"Médias:")
print(f"  AUC    — orig: {df_res['AUC_orig'].mean():.4f}  cal: {df_res['AUC_cal'].mean():.4f}  "
      #delta AUC é pequeno, mas positivo, indicando que a calibração não prejudicou a discriminação do modelo
      f"(Δ = {df_res['AUC_cal'].mean() - df_res['AUC_orig'].mean():+.4f})")
print(f"  Brier  — orig: {df_res['Brier_orig'].mean():.4f}  cal: {df_res['Brier_cal'].mean():.4f}  "
#delta Brier é negativo, indicando que a calibração melhorou a qualidade das probabilidades previstas, mesmo que a AUC tenha permanecido similar
      f"(Δ = {df_res['delta_brier'].mean():+.4f})")

print(f"\nModelos calibrados salvos em: {OUTPUT_DIR}")

df_res.to_csv('../data/gold/metricas_calibracao.csv', index=False)
