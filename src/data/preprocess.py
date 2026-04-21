"""
preprocess.py
Limpieza y preparacion del dataset de potabilidad del agua.
"""

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_PATH = BASE_DIR / "data/raw/water_potability.csv"
PROCESSED_PATH = BASE_DIR / "data/processed/water_potability_clean.csv"
TARGET_COLUMN = "Potability"
FEATURE_COLUMNS = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Dataset cargado desde {path}: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Muestra informacion basica del dataset."""
    print("\n--- Primeras filas ---")
    print(df.head().to_string())
    print("\n--- Tipos de datos ---")
    print(df.dtypes.to_string())
    print("\n--- Valores nulos ---")
    print(df.isnull().sum().to_string())
    print("\n--- Duplicados ---")
    print(df.duplicated().sum())
    print("\n--- Estadisticas descriptivas ---")
    print(df.describe().to_string())
    print("\n--- Distribucion de la variable objetivo ---")
    print(df[TARGET_COLUMN].value_counts().to_string())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza basica:
    - Elimina duplicados.
    - Rellena nulos de variables numericas con la mediana.
    """
    clean_df = df.drop_duplicates().copy()

    numeric_columns = clean_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_numeric_columns = [col for col in numeric_columns if col != TARGET_COLUMN]

    for col in feature_numeric_columns:
        clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    print(f"\nDataset limpio: {clean_df.shape[0]} filas, {clean_df.shape[1]} columnas")
    print("\n--- Nulos despues de limpiar ---")
    print(clean_df.isnull().sum().to_string())
    return clean_df


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa variables de entrada y salida."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    print(f"\nEntradas ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
    print(f"Salida: {TARGET_COLUMN}")
    return X, y


def save_processed_data(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nDatos procesados guardados en: {path.resolve()}")


if __name__ == "__main__":
    df = load_data(RAW_PATH)
    inspect_data(df)
    clean_df = clean_data(df)
    X, y = select_features(clean_df)

    output_df = X.copy()
    output_df[TARGET_COLUMN] = y
    save_processed_data(output_df, PROCESSED_PATH)
