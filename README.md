# Sistema de Monitoreo de Calidad del Agua

Modelo de Machine Learning con Random Forest para estimar la potabilidad del agua
usando variables fisicoquímicas medidas por sensores y laboratorio.

## Variables

**Entrada:**

- `ph`
- `Hardness`
- `Solids`
- `Chloramines`
- `Sulfate`
- `Conductivity`
- `Organic_carbon`
- `Trihalomethanes`
- `Turbidity`

**Salida:**

- `Potability`

## Estructura del proyecto

```text
RF-IA-water-detection-model/
|-- data/
|   |-- raw/            # Dataset original de Kaggle
|   `-- processed/      # Datos limpios y listos para entrenar
|-- models/             # Modelo entrenado guardado (.pkl)
|-- notebooks/          # Exploración y análisis del dataset
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

## Instalación

```bash
pip install -r requirements.txt
```

## Configurar Jupyter y Entorno Virtual

1. Instalar la extensión de Jupyter en VSCode:
   [Jupyter Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

2. Instalar Jupyter en Python:

```bash
pip install jupyter notebook ipykernel
```

3. Seleccionar un Kernel:
   - Abre o crea un archivo `.ipynb`
   - En la esquina superior derecha haz clic en **"Select Kernel"**
   - Elige tu intérprete de Python o entorno virtual

## Uso rápido

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