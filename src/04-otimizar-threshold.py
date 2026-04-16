import pandas as pd
import numpy as np
import joblib
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_recall_curve, roc_auc_score,
                             precision_score, recall_score, f1_score)

# Uso: python 04-otimizar-threshold.py [periodos] [rf|xgb]
# Ex:  python 04-otimizar-threshold.py 40,60,90 xgb
args = [a for a in sys.argv[1:] if a]
periodos_arg = next((a for a in args if a[0].isdigit()), None)
modelo_tipo  = next((a for a in args if a in ('rf', 'xgb')), 'rf')
periodos = [int(d) for d in periodos_arg.split(',')] if periodos_arg else [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]

MODELS_DIR = f'../models/{modelo_tipo}'
OUTPUT_CSV = f'../data/gold/thresholds_{modelo_tipo}.csv'
RECALL_MINIMO = 0.50

feature_names = ['idade_dias', 'num_manutencoes', 'intervalo_medio_manut',
                 'num_falhas_historico', 'taxa_falhas_ano', 'minutos_falha_historico',
                 'taxa_minutos_falha_ano', 'dias_desde_ultima_falha', 'dias_desde_ultima_manut', 'limite_potencia',
                 'utilizacao_media', 'utilizacao_maxima', 'utilizacao_minima',
                 'utilizacao_desvio', 'taxa_sobrecargas_ano',
                 'p90_utilizacao', 'delta_utilizacao', 'utilizacao_tendencia_90d', 'dias_acima_80pct_limite']

resultados = []

print(f"{'='*95}")
print(f"{'Dias':>4}  {'Thr padrão':>10}  {'Rec 0.5':>7}  {'Prec 0.5':>8}  {'F1 0.5':>6}  "
      f"{'Thr F1-max':>10}  {'Rec F1':>7}  {'Prec F1':>8}  {'F1 F1':>6}")
print(f"{'='*95}")

for periodo in periodos:
    modelo_path = f'{MODELS_DIR}/preditivo_{periodo}d.pkl'
    features_path = f'../data/gold/features_preditivo_{periodo}d.csv'

    try:
        modelo = joblib.load(modelo_path)
        df = pd.read_csv(features_path)
    except FileNotFoundError as e:
        print(f"[{periodo}d] Arquivo não encontrado: {e}")
        continue

    df = df[df['dias_com_dados_util'] > 0]
    X = df[feature_names]
    y = df['vai_falhar']

    # Reproduz exatamente o mesmo split do treino
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    y_proba = modelo.predict_proba(X_test)[:, 1]

    # --- Threshold padrão (0.5) ---
    y_pred_default = (y_proba >= 0.5).astype(int)
    rec_default = recall_score(y_test, y_pred_default, zero_division=0)
    prec_default = precision_score(y_test, y_pred_default, zero_division=0)
    f1_default = f1_score(y_test, y_pred_default, zero_division=0)

    precision_curve, recall_curve, thresholds_curve = precision_recall_curve(y_test, y_proba)

    # --- Estratégia 1: recall mínimo de RECALL_MINIMO ---
    candidatos_recall = np.where(recall_curve[:-1] >= RECALL_MINIMO)[0]
    if len(candidatos_recall) > 0:
        idx_recall = candidatos_recall[np.argmax(precision_curve[candidatos_recall])]
        thr_recall = thresholds_curve[idx_recall]
    else:
        # Se não há threshold que atinja o recall mínimo, pega o de menor threshold
        thr_recall = thresholds_curve[0]

    y_pred_recall = (y_proba >= thr_recall).astype(int)
    rec_r = recall_score(y_test, y_pred_recall, zero_division=0)
    prec_r = precision_score(y_test, y_pred_recall, zero_division=0)
    f1_r = f1_score(y_test, y_pred_recall, zero_division=0)

    # --- Estratégia 2: maximizar F1 ---
    f1_scores = (2 * precision_curve[:-1] * recall_curve[:-1] /
                 (precision_curve[:-1] + recall_curve[:-1] + 1e-9))
    idx_f1 = np.argmax(f1_scores)
    thr_f1 = thresholds_curve[idx_f1]

    y_pred_f1 = (y_proba >= thr_f1).astype(int)
    rec_f = recall_score(y_test, y_pred_f1, zero_division=0)
    prec_f = precision_score(y_test, y_pred_f1, zero_division=0)
    f1_f = f1_score(y_test, y_pred_f1, zero_division=0)

    print(f"{periodo:>4}  "
          f"0.500 → R={rec_default:.2f} P={prec_default:.2f} F1={f1_default:.2f}  |  "
          f"{thr_recall:.3f} → R={rec_r:.2f} P={prec_r:.2f} F1={f1_r:.2f}  |  "
          f"{thr_f1:.3f} → R={rec_f:.2f} P={prec_f:.2f} F1={f1_f:.2f}")

    resultados.append({
        'periodo_dias': periodo,
        'threshold_padrao': 0.5,
        'recall_padrao': round(rec_default, 4),
        'precision_padrao': round(prec_default, 4),
        'f1_padrao': round(f1_default, 4),
        'threshold_recall_minimo': round(float(thr_recall), 4),
        'recall_recall_minimo': round(rec_r, 4),
        'precision_recall_minimo': round(prec_r, 4),
        'f1_recall_minimo': round(f1_r, 4),
        'threshold_f1_max': round(float(thr_f1), 4),
        'recall_f1_max': round(rec_f, 4),
        'precision_f1_max': round(prec_f, 4),
        'f1_f1_max': round(f1_f, 4),
    })

print(f"{'='*95}")

df_out = pd.DataFrame(resultados)
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\nThresholds salvos em: {OUTPUT_CSV}")

print(f"\n{'='*60}")
print("GANHO MÉDIO COM THRESHOLD OTIMIZADO (F1-max vs padrão)")
print(f"{'='*60}")
print(f"Recall:    {df_out['recall_padrao'].mean():.3f} → {df_out['recall_f1_max'].mean():.3f}  "
      f"(+{(df_out['recall_f1_max'].mean() - df_out['recall_padrao'].mean()):.3f})")
print(f"Precision: {df_out['precision_padrao'].mean():.3f} → {df_out['precision_f1_max'].mean():.3f}  "
      f"({(df_out['precision_f1_max'].mean() - df_out['precision_padrao'].mean()):+.3f})")
print(f"F1-Score:  {df_out['f1_padrao'].mean():.3f} → {df_out['f1_f1_max'].mean():.3f}  "
      f"(+{(df_out['f1_f1_max'].mean() - df_out['f1_padrao'].mean()):.3f})")
