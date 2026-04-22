"""
evaluate.py
Evaluacion del modelo entrenado: metricas, matriz de confusion e importancia de variables.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models/random_forest.pkl"
REPORTS_DIR = BASE_DIR / "reports"
CLASS_NAMES = ["No potable", "Potable"]


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    X_test = pd.read_csv(BASE_DIR / "data/processed/X_test.csv")
    y_test = pd.read_csv(BASE_DIR / "data/processed/y_test.csv").squeeze()
    return X_test, y_test


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("=" * 50)
    print("RESULTADOS DEL MODELO")
    print("=" * 50)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
    print("\nReporte completo:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    return y_pred, y_prob


def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusion - Potabilidad del agua")
    plt.tight_layout()
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(path, dpi=150)
    print(f"Grafico guardado: {path}")
    plt.show()


def plot_feature_importance(model, feature_names: list[str]) -> None:
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(
        [feature_names[i] for i in indices],
        importances[indices],
        color="#3B8BD4",
    )
    ax.set_xlabel("Importancia")
    ax.set_title("Importancia de variables - Random Forest")
    ax.invert_yaxis()

    for bar, val in zip(bars, importances[indices]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center")

    plt.tight_layout()
    path = REPORTS_DIR / "feature_importance.png"
    plt.savefig(path, dpi=150)
    print(f"Grafico guardado: {path}")
    plt.show()


def plot_roc_curve(y_test: pd.Series, y_prob: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1D9E75", lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curva ROC - Potabilidad del agua")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = REPORTS_DIR / "roc_curve.png"
    plt.savefig(path, dpi=150)
    print(f"Grafico guardado: {path}")
    plt.show()


if __name__ == "__main__":
    model = joblib.load(MODEL_PATH)
    X_test, y_test = load_test_data()
    y_pred, y_prob = evaluate(model, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(model, list(X_test.columns))
    plot_roc_curve(y_test, y_prob)
