# 1. Importações
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# 2. Carga e Preparação (Mantendo a transformação logarítmica)
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

df["Amount_log"] = np.log1p(df["Amount"])
df = df.drop("Amount", axis=1)

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# 3. Treinamento Inicial com XGBoost
print("\n--- Treinando XGBoost Padrão ---")
xgb = XGBClassifier(
    scale_pos_weight=10, 
    eval_metric="logloss",
    random_state=42
)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
print(classification_report(y_test, y_pred_xgb))

# 4. Feature Importance (Gráfico simples do XGBoost)
importancias = xgb.feature_importances_
plt.figure(figsize=(10, 5))
plt.bar(range(len(importancias)), importancias)
plt.title("Importância das Variáveis (Visão Geral)")
plt.show()

# 5. Ajuste Fino de Parâmetros (GridSearchCV)
print("\n--- Buscando os melhores parâmetros (Aguarde...) ---")
param_grid = {
    "max_depth": [3, 5],
    "n_estimators": [50, 100]
}

# Usando n_jobs=-1 para acelerar a busca usando todos os núcleos do PC
grid = GridSearchCV(
    XGBClassifier(scale_pos_weight=10, eval_metric="logloss", random_state=42),
    param_grid,
    scoring="recall",
    cv=3,
    n_jobs=-1 
)

grid.fit(X_train, y_train)
print("Melhor combinação encontrada:", grid.best_params_)

# Avaliando o melhor modelo encontrado pelo GridSearch
melhor_modelo = grid.best_estimator_
y_pred_grid = melhor_modelo.predict(X_test)
print("\n--- Relatório após o GridSearch ---")
print(classification_report(y_test, y_pred_grid))

# 6. Explicabilidade com SHAP
print("\n--- Gerando explicação SHAP ---")
explainer = shap.Explainer(melhor_modelo)
# Calculando SHAP values apenas para as 100 primeiras linhas para economizar tempo
shap_values = explainer(X_test[:100])

shap.plots.bar(shap_values)