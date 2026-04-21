"""
predict.py
Realiza predicciones con el modelo entrenado sobre nuevas muestras.
"""

from pathlib import Path

import joblib
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models/random_forest.pkl"
SCALER_PATH = BASE_DIR / "models/scaler.pkl"
FEATURE_NAMES = [
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


def predict_sample(**sample_values: float) -> dict:
    """
    Retorna:
    - prediction: 0 = no potable, 1 = potable
    - label: texto legible
    - probability: probabilidad de ser potable
    """
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    sample = np.array([[sample_values[name] for name in FEATURE_NAMES]])
    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)[0]
    probability = model.predict_proba(sample_scaled)[0][1]
    label = "POTABLE" if prediction == 1 else "NO POTABLE"

    return {
        "prediction": int(prediction),
        "label": label,
        "probability": round(float(probability), 4),
    }


if __name__ == "__main__":
    samples = [
        {
            "ph": 7.0,
            "Hardness": 200.0,
            "Solids": 20000.0,
            "Chloramines": 7.0,
            "Sulfate": 330.0,
            "Conductivity": 420.0,
            "Organic_carbon": 14.0,
            "Trihalomethanes": 66.0,
            "Turbidity": 4.0,
        },
        {
            "ph": 4.5,
            "Hardness": 260.0,
            "Solids": 38000.0,
            "Chloramines": 10.0,
            "Sulfate": 390.0,
            "Conductivity": 600.0,
            "Organic_carbon": 20.0,
            "Trihalomethanes": 95.0,
            "Turbidity": 5.8,
        },
    ]

    print("=" * 55)
    print("PREDICCIONES - SISTEMA DE CALIDAD DEL AGUA")
    print("=" * 55)

    for i, sample in enumerate(samples, 1):
        result = predict_sample(**sample)
        print(f"\nMuestra {i}:")
        print(f"  Entrada  -> {sample}")
        print(f"  Resultado -> {result['label']}")
        print(f"  Probabilidad de ser potable: {result['probability'] * 100:.1f}%")
