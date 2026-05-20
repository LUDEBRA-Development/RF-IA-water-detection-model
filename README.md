# Sistema de Monitoreo de Calidad del Agua

Modelo de Machine Learning con Random Forest para estimar la potabilidad del agua
usando solo las variables de sensores definidas en el proyecto.

## Variables

**Entrada:**

- `ph`
- `Turbidity`
- `Conductivity`
- `Solids`

**Salida:**

- `Potability`

## Estado del dataset

El preprocesamiento parte de `data/raw/water_potability.csv` y deja el dataset listo
para entrenar con `ph`, `Turbidity`, `Conductivity` y `Solids`.

El sistema fue disenado para soportar sensores de temperatura, pero el dataset
utilizado no contiene esa variable, por lo que el modelo actual se entrena con
pH y turbidez junto a otras variables fisicoquimicas disponibles.

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

## Configurar Jupyter y Entorno Virtual

1. Instalar la extension de Jupyter en VSCode:
   [Jupyter Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

2. Instalar Jupyter en Python:

```bash
pip install jupyter notebook ipykernel
```

3. Seleccionar un Kernel:
   - Abre o crea un archivo `.ipynb`
   - En la esquina superior derecha haz clic en **"Select Kernel"**
   - Elige tu interprete de Python o entorno virtual

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
