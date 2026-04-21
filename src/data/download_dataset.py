"""
download_dataset.py
Descarga el dataset de potabilidad del agua desde Kaggle y lo copia al workspace.
"""

import shutil
from pathlib import Path

import kagglehub


BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_HANDLE = "adityakadiwal/water-potability"
SOURCE_FILENAME = "water_potability.csv"
DESTINATION_PATH = BASE_DIR / "data/raw/water_potability.csv"


def download_dataset() -> Path:
    dataset_dir = Path(kagglehub.dataset_download(DATASET_HANDLE))
    source_path = dataset_dir / SOURCE_FILENAME

    if not source_path.exists():
        raise FileNotFoundError(f"No se encontro el archivo esperado: {source_path}")

    DESTINATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, DESTINATION_PATH)
    return DESTINATION_PATH


if __name__ == "__main__":
    saved_path = download_dataset()
    print(f"Dataset guardado en: {saved_path.resolve()}")
