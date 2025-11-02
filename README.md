# scikit-learn-regresion
Pequeño proyecto de regresión con scikit-learn para predecir el valor medio de la vivienda en distritos de California usando un RandomForestRegressor. Incluye grid search, métricas básicas y visualizaciones.

# Objetivo

Entrenar un modelo que prediga MedHouseVal y evaluar su rendimiento (MAE, RMSE, R²), además de interpretar qué variables aportan más.

# Dataset

California Housing (fetch_california_housing).

Variables: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude.

Nota: el objetivo viene recortado a 5.0 (hay muchos valores en el tope), lo que genera más error en la cola alta.

# Metodología

Split: 80% train / 20% test (random_state=42).

Modelo: RandomForestRegressor (sin escalado: los árboles no lo necesitan).

Búsqueda de hiperparámetros: GridSearchCV (varias combinaciones de n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features).

# Resultados

Mejores parámetros:
n_estimators=200, max_depth=20, min_samples_split=2, min_samples_leaf=1, max_features='sqrt'.

CV (RMSE): 0.510

Test:

MAE: 0.327

RMSE: 0.504

R²: 0.806
(Escala del target ≈ cientos de miles de $ → RMSE ≈ 50k$ aprox.)

# Gráficas

Predicción vs Real: nube cerca de la diagonal; columna en x≈5 por el clipping del objetivo.

Distribución de residuales: centrada en 0; algo más de dispersión en precios altos.


# Top features en mi run:

MedInc — 0.527

AveOccup — 0.138

Latitude — 0.089

Longitude — 0.088

HouseAge — 0.054
(Otras: AveRooms, Population, AveBedrms)

# Cómo ejecutar

python -m venv .venv
.venv\Scripts\activate

# ejecutar
python main.py
python mainV2.py
