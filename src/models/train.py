"""
train.py
Entrenamiento del modelo Random Forest para detección de contaminación.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# Rutas
PROCESSED_PATH = Path("data/processed/water_clean.csv")
MODEL_PATH     = Path("models/random_forest.pkl")
SCALER_PATH    = Path("models/scaler.pkl")

# Columna objetivo (debe coincidir con preprocess.py)
TARGET_COLUMN  = "Potability"


def load_processed_data():
    df = pd.read_csv(PROCESSED_PATH)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Divide los datos: 80% entrenamiento, 20% prueba.
    random_state=42 asegura que siempre obtengas el mismo split (reproducibilidad).
    stratify=y mantiene la proporción de clases en ambos conjuntos.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Entrenamiento: {X_train.shape[0]} muestras")
    print(f"Prueba:        {X_test.shape[0]} muestras")
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Normaliza las features con StandardScaler.
    IMPORTANTE: fit solo en X_train, transform en ambos.
    Esto evita 'data leakage' (filtrar info del test al entrenamiento).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train, y_train):
    """
    Crea y entrena el modelo Random Forest.

    Hiperparámetros:
      n_estimators : número de árboles (100 es un buen punto de partida)
      max_depth    : profundidad máxima de cada árbol (None = sin límite)
      random_state : semilla para reproducibilidad
      n_jobs       : núcleos de CPU a usar (-1 = todos)
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )

    print("\nEntrenando Random Forest...")
    model.fit(X_train, y_train)
    print("Entrenamiento completado.")

    # Validación cruzada (5 folds) — estimación más confiable del rendimiento
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
    print(f"\nValidación cruzada F1 (5 folds): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return model


def save_artifacts(model, scaler):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nModelo guardado en:  {MODEL_PATH}")
    print(f"Scaler guardado en:  {SCALER_PATH}")


if __name__ == "__main__":
    X, y = load_processed_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)
    model = train_model(X_train_s, y_train)
    save_artifacts(model, scaler)

    # Guardar X_test e y_test para usar en evaluate.py
    pd.DataFrame(X_test_s, columns=X.columns).to_csv("data/processed/X_test.csv", index=False)
    pd.Series(y_test.values, name=TARGET_COLUMN).to_csv("data/processed/y_test.csv", index=False)
    print("\nDatos de prueba guardados para evaluación.")
