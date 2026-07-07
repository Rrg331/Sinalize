import pandas as pd
import numpy as np
import joblib
import sys
import gc
import os
from xgboost import XGBClassifier
from util.config import PERIODOS, FEATURE_NAMES
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             recall_score, precision_score, classification_report)

periodos = [int(d) for d in sys.argv[1].split(',')] if len(sys.argv) > 1 else PERIODOS
feature_names = FEATURE_NAMES

MODELS_DIR = '../models/xgb'
os.makedirs(MODELS_DIR, exist_ok=True)

resultados = []
inicio = pd.Timestamp.now()

for periodo in periodos:
    print(f"\nTreinando XGBoost para {periodo} dias no futuro")

    df = pd.read_csv(f'../data/gold/features_preditivo_{periodo}d.csv')
    df = df[df['dias_com_dados_util'] > 0]

    X = df[feature_names]
    y = df['vai_falhar']

    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_pos_weight = n_neg / n_pos  # compensa desbalanceamento sem SMOTE
    print(f"Distribuição: Falhas={n_pos} ({n_pos/len(y)*100:.1f}%), OK={n_neg} — scale_pos_weight={scale_pos_weight:.1f}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1],
    }

    xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        verbosity=0,
        use_label_encoder=False
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(xgb, param_grid, cv=cv, scoring='roc_auc', n_jobs=1, verbose=0)
    grid.fit(X_train, y_train)

    model = grid.best_estimator_
    print(f"Melhores params: {grid.best_params_}")

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = model.predict(X_test)

    auc  = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    acc  = accuracy_score(y_test, y_pred)

    print(f"ROC-AUC: {auc:.3f}  Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}")

    resultados.append({
        'Dias': periodo,
        'ROC-AUC': auc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'Accuracy': acc,
        'Falhas': y.sum(),
        '% Falhas': y.sum() / len(y) * 100
    })

    joblib.dump(model, f'{MODELS_DIR}/preditivo_{periodo}d.pkl')

    del model, grid, X, y, X_train, X_test, y_train, y_test, df
    gc.collect()

print(f"\n{'='*70}")
print("RESULTADO XGBoost")
print('='*70)
df_res = pd.DataFrame(resultados)
print(df_res[['Dias', 'ROC-AUC', 'Precision', 'Recall', 'F1-Score']].to_string(
    index=False, float_format=lambda x: f'{x:.3f}'))
print(f"\nAUC médio XGBoost: {df_res['ROC-AUC'].mean():.4f}")
print(f"Tempo total: {pd.Timestamp.now() - inicio}")

# Salva métricas XGBoost para comparação
df_res.to_csv('../data/gold/metricas_xgboost.csv', index=False)
print("\nMétricas salvas em: ../data/gold/metricas_xgboost.csv")

print(f"Modelos XGBoost salvos em: {MODELS_DIR}")
