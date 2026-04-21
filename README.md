# Sistema de Monitoreo de Calidad del Agua

Modelo de Machine Learning con Random Forest para estimar la potabilidad del agua
usando variables fisicoquimicas medidas por sensores y laboratorio.

Variables de entrada:

- `ph`
- `Hardness`
- `Solids`
- `Chloramines`
- `Sulfate`
- `Conductivity`
- `Organic_carbon`
- `Trihalomethanes`
- `Turbidity`

Variable de salida:

- `Potability`

## Estructura del proyecto

```text
RF-IA-water-detection-model/
|-- data/
|   |-- raw/            # Dataset original de Kaggle
|   `-- processed/      # Datos limpios y listos para entrenar
|-- models/             # Modelo entrenado guardado (.pkl)
|-- notebooks/          # Exploracion y analisis del dataset
|-- src/
|   |-- data/
|   |   |-- download_dataset.py
|   |   `-- preprocess.py
|   `-- models/
|       |-- train.py
|       |-- evaluate.py
|       `-- predict.py
|-- requirements.txt
`-- README.md
```

## Instalacion

```bash
pip install -r requirements.txt
```

## Uso rapido

```bash
# 1. Descargar el dataset
python src/data/download_dataset.py

# 2. Preprocesar los datos
python src/data/preprocess.py

# 3. Entrenar el modelo
python src/models/train.py

# 4. Evaluar resultados
python src/models/evaluate.py

# 5. Predecir una nueva muestra
python src/models/predict.py
```
