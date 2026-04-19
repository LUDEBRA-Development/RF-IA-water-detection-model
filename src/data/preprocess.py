"""
preprocess.py
Limpieza y preparación del dataset de calidad del agua.
Ajusta las columnas según el dataset que descargues de Kaggle.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Rutas
RAW_PATH       = Path("data/raw/water_quality.csv")   # <-- cambia por tu archivo
PROCESSED_PATH = Path("data/processed/water_clean.csv")


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(df.head())
    return df


def inspect_data(df: pd.DataFrame):
    """Muestra información básica: tipos, nulos, estadísticas."""
    print("\n--- Tipos de datos ---")
    print(df.dtypes)
    print("\n--- Valores nulos ---")
    print(df.isnull().sum())
    print("\n--- Estadísticas descriptivas ---")
    print(df.describe())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza básica:
    - Elimina duplicados
    - Rellena valores nulos con la mediana de cada columna
    """
    df = df.drop_duplicates()

    # Rellenar nulos con la mediana (más robusto que la media para datos de sensores)
    for col in df.select_dtypes(include=[np.number]).columns:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    print(f"\nDataset limpio: {df.shape[0]} filas")
    return df


def select_features(df: pd.DataFrame, feature_cols: list, target_col: str) -> tuple:
    """
    Selecciona las columnas de sensores (features) y la columna objetivo (target).

    Parámetros:
        feature_cols : lista con los nombres de las columnas de sensores
                       Ejemplo: ['ph', 'Turbidity', 'Temperature']
        target_col   : nombre de la columna con la etiqueta
                       Ejemplo: 'Potability' o 'is_safe'
    """
    X = df[feature_cols]
    y = df[target_col]
    print(f"\nFeatures: {feature_cols}")
    print(f"Target: {target_col} | Distribución:\n{y.value_counts()}")
    return X, y


if __name__ == "__main__":
    # ---------------------------------------------------------------
    # AJUSTA ESTAS VARIABLES según las columnas de tu dataset Kaggle
    # ---------------------------------------------------------------
    FEATURE_COLUMNS = ["ph", "Turbidity", "Temperature"]  # <-- nombres reales del CSV
    TARGET_COLUMN   = "Potability"                         # <-- columna de etiqueta

    df = load_data(RAW_PATH)
    inspect_data(df)
    df = clean_data(df)

    X, y = select_features(df, FEATURE_COLUMNS, TARGET_COLUMN)

    # Guardar datos procesados
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df = X.copy()
    clean_df[TARGET_COLUMN] = y.values
    clean_df.to_csv(PROCESSED_PATH, index=False)
    print(f"\nDatos guardados en: {PROCESSED_PATH}")
