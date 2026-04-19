"""
predict.py
Realiza predicciones con el modelo entrenado sobre nuevas muestras de sensores.
Simula los valores que llegarían de sensores físicos en tiempo real.
"""

import numpy as np
import joblib
from pathlib import Path

MODEL_PATH  = Path("models/random_forest.pkl")
SCALER_PATH = Path("models/scaler.pkl")

# Nombres de las features (deben coincidir con el orden del entrenamiento)
FEATURE_NAMES = ["ph", "Turbidity", "Temperature"]


def predict_sample(ph: float, turbidity: float, temperature: float) -> dict:
    """
    Recibe los valores de los tres sensores y devuelve la predicción.

    Retorna un dict con:
        - prediction : 0 = potable, 1 = no potable
        - label      : texto legible
        - probability: probabilidad de contaminación (0.0 a 1.0)
    """
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    sample = np.array([[ph, turbidity, temperature]])
    sample_scaled = scaler.transform(sample)

    prediction  = model.predict(sample_scaled)[0]
    probability = model.predict_proba(sample_scaled)[0][1]

    label = "NO POTABLE ⚠️" if prediction == 1 else "POTABLE ✓"

    return {
        "prediction":  int(prediction),
        "label":       label,
        "probability": round(float(probability), 4)
    }


if __name__ == "__main__":
    # Ejemplos de prueba — ajusta los valores según los rangos de tu dataset
    samples = [
        {"ph": 7.0,  "turbidity": 2.1,  "temperature": 22.0},  # agua normal
        {"ph": 4.2,  "turbidity": 15.8, "temperature": 31.0},  # posible contaminación
        {"ph": 8.5,  "turbidity": 0.8,  "temperature": 18.0},  # agua limpia
    ]

    print("=" * 55)
    print("PREDICCIONES — SISTEMA DE CALIDAD DEL AGUA")
    print("=" * 55)

    for i, s in enumerate(samples, 1):
        result = predict_sample(s["ph"], s["turbidity"], s["temperature"])
        print(f"\nMuestra {i}:")
        print(f"  Entrada  → pH: {s['ph']} | Turbidez: {s['turbidity']} | Temp: {s['temperature']}°C")
        print(f"  Resultado → {result['label']}")
        print(f"  Probabilidad de contaminación: {result['probability']*100:.1f}%")
