# Sistema de Monitoreo de Calidad del Agua

Modelo de Machine Learning con Random Forest para detectar contaminación en fuentes hídricas
usando datos de sensores de pH, turbidez y temperatura.

- pH
- turbidez
- temperatura

## Estructura del proyecto

```
water_quality_monitor/
├── data/
│   ├── raw/            # Dataset original de Kaggle (sin modificar)
│   └── processed/      # Datos limpios y listos para entrenar
├── models/             # Modelo entrenado guardado (.pkl)
├── notebooks/          # Exploración y análisis del dataset
├── src/
│   ├── preprocess.py   # Limpieza y preparación de datos
│   ├── train.py        # Entrenamiento del modelo
│   ├── evaluate.py     # Evaluación y métricas
│   └── predict.py      # Predicciones con nuevas muestras
├── reports/            # Gráficos y resultados exportados
├── requirements.txt
└── README.md
```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso rápido

```bash
# 1. Preprocesar los datos
python src/preprocess.py

# 2. Entrenar el modelo
python src/train.py

# 3. Evaluar resultados
python src/evaluate.py

# 4. Predecir una nueva muestra
python src/predict.py
```
