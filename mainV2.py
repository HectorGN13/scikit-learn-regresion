import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1) Datos
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2) Pipeline (solo modelo, sin escalado)
pipe = Pipeline([
    ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
])

# 3) Grid de hiperparámetros (más completo para RF)
param_grid = {
    "model__n_estimators": [200],
    "model__max_depth": [12, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
    "model__max_features": ["sqrt", None]  # None = usar todas
}

gs = GridSearchCV(
    pipe, param_grid,
    cv=3, n_jobs=-1,
    scoring="neg_root_mean_squared_error",
    verbose=0
)
gs.fit(X_train, y_train)

print("Mejores parámetros:", gs.best_params_)
print("Mejor RMSE CV:", -gs.best_score_)

# 4) Evaluación en test
best = gs.best_estimator_
y_pred = best.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

try:
    rmse = mean_squared_error(y_test, y_pred, squared=False)
except TypeError:
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R^2:  {r2:.3f}")

# 5) Gráfico Predicción vs Real
plt.figure()
plt.scatter(y_test, y_pred, edgecolor="k", alpha=0.7)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, "--", linewidth=2)
plt.xlabel("Valor real")
plt.ylabel("Predicción")
plt.title("Predicción vs Real")
plt.tight_layout()
plt.show()

# 6) Distribución de residuales
resid = y_test - y_pred
plt.figure()
plt.hist(resid, bins=30)
plt.title("Distribución de residuales")
plt.xlabel("Residual")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()

# 7) Importancias de variables
rf = best.named_steps["model"]
imp = rf.feature_importances_
feat_names = np.array(X.columns)

order = np.argsort(imp)[::-1]
print("\nImportancias (RandomForest):")
for i in order:
    print(f"{feat_names[i]:<15} {imp[i]:.3f}")
