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
SELECTED_FEATURE_COLUMNS = ["ph", "Turbidity", "Conductivity", "Solids"]
MISSING_PROJECT_SENSOR = "temperatura"


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
    Limpieza del dataset:
    - Elimina duplicados.
    - Rellena nulos numericos con la mediana.
    - Recorta valores extremos con el metodo IQR.
    """
    clean_df = df.drop_duplicates().copy()

    numeric_columns = clean_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_numeric_columns = [col for col in numeric_columns if col != TARGET_COLUMN]

    for col in feature_numeric_columns:
        clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    outlier_summary = {}
    for col in SELECTED_FEATURE_COLUMNS:
        q1 = clean_df[col].quantile(0.25)
        q3 = clean_df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = ((clean_df[col] < lower_bound) | (clean_df[col] > upper_bound)).sum()
        outlier_summary[col] = int(outlier_count)
        clean_df[col] = clean_df[col].clip(lower=lower_bound, upper=upper_bound)

    print(f"\nDataset limpio: {clean_df.shape[0]} filas, {clean_df.shape[1]} columnas")
    print("\n--- Nulos despues de limpiar ---")
    print(clean_df.isnull().sum().to_string())
    print("\n--- Valores extremos recortados (IQR) ---")
    print(pd.Series(outlier_summary).to_string())
    return clean_df


def select_project_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Conserva las variables mas utiles del dataset para el proyecto actual.
    El modelo sigue sin temperatura porque el dataset original no la contiene.
    """
    X = df[SELECTED_FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    print(f"\nEntradas seleccionadas ({len(SELECTED_FEATURE_COLUMNS)}): {SELECTED_FEATURE_COLUMNS}")
    print(f"Salida: {TARGET_COLUMN}")
    print(
        "\nNota metodologica: el sistema fue pensado para soportar temperatura, "
        f"pero el dataset original no contiene esa variable ({MISSING_PROJECT_SENSOR}). "
        "Por eso el modelo actual usa pH, turbidez, conductividad y solidos disueltos."
    )

    return X, y


def save_processed_data(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nDatos procesados guardados en: {path.resolve()}")


if __name__ == "__main__":
    df = load_data(RAW_PATH)
    inspect_data(df)
    clean_df = clean_data(df)
    X, y = select_project_features(clean_df)

    output_df = X.copy()
    output_df[TARGET_COLUMN] = y
    save_processed_data(output_df, PROCESSED_PATH)
