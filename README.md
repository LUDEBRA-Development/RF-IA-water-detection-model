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
# 1. Instalar los Requerimientos
pip install -r requirements.txt
```

## Configurar Jupyter y Entorno Virtual (venv)

1. Instalar Extension de Jupyter en VSCode

```bash
https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter
```

2. Instalar Jupyter en Python

```bash
pip install jupyter notebook ipykernel
```

3. Seleccionar un Kernel (Entorno de Pyton)
   - Abre o crea un archivo .ipynb
   - En la esquina superior derecha verás "Select Kernel"
   - Elige tu intérprete de Python (o un entorno virtual si usas uno)

## Uso rápido (RAW)

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
