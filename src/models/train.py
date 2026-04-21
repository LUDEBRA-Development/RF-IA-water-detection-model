"""
train.py
Entrenamiento del modelo Random Forest para estimar la potabilidad del agua.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_PATH = BASE_DIR / "data/processed/water_potability_clean.csv"
MODEL_PATH = BASE_DIR / "models/random_forest.pkl"
SCALER_PATH = BASE_DIR / "models/scaler.pkl"
TARGET_COLUMN = "Potability"


def load_processed_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(PROCESSED_PATH)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"Entrenamiento: {X_train.shape[0]} muestras")
    print(f"Prueba:        {X_test.shape[0]} muestras")
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    print("\nEntrenando Random Forest...")
    model.fit(X_train, y_train)
    print("Entrenamiento completado.")

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
    print(f"\nValidacion cruzada F1 (5 folds): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    return model


def save_artifacts(model: RandomForestClassifier, scaler: StandardScaler) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nModelo guardado en: {MODEL_PATH}")
    print(f"Scaler guardado en: {SCALER_PATH}")


if __name__ == "__main__":
    X, y = load_processed_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    model = train_model(X_train_scaled, y_train)
    save_artifacts(model, scaler)

    processed_dir = PROCESSED_PATH.parent
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(processed_dir / "X_test.csv", index=False)
    pd.Series(y_test.values, name=TARGET_COLUMN).to_csv(processed_dir / "y_test.csv", index=False)
    print("\nDatos de prueba guardados para evaluacion.")
