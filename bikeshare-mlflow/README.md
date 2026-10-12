# Experimentos de demanda de bicicletas con MLflow

Proyecto generado con Cookiecutter para llevar el modelo de SeoulBikeData del
notebook exploratorio a ejecuciones CLI reproducibles. Conserva la regresión
lineal original como referencia y registra Random Forest parametrizados en un
experimento MLflow independiente.

## Organización del código

- `run_experiment.py`: punto de entrada CLI y presentación del resultado.
- `bikeshare/data_loading.py`: lectura del CSV y validación de columnas.
- `bikeshare/cleaning.py`: conversiones, eliminación de duplicados y filas inválidas.
- `bikeshare/preprocessing.py`: variables de calendario, split cronológico e imputación/codificación.
- `bikeshare/modeling.py`: ajuste del Random Forest y la regresión lineal de referencia.
- `bikeshare/evaluation.py`: métricas y artefactos gráficos/JSON.
- `bikeshare/tracking.py`: experimento, parámetros, métricas, artefactos y modelo MLflow.
- `bikeshare/pipeline.py`: coordina las etapas sin concentrar su lógica.
- `bikeshare/settings.py` y `bikeshare/cli.py`: configuración común y argumentos CLI.
- `run_experiments.py`: ejecuta los cuatro barridos de cinco niveles.

## Requisitos y datos

- Python 3.14.7 (versión usada para estas corridas).
- `pip install -r requirements.txt`.
- El CSV `data/SeoulBikeData.csv`, leído con codificación `cp1252`.

## Una corrida

Desde esta carpeta, sin argumentos:

```powershell
python run_experiment.py
```

Los valores por defecto son 120 árboles, profundidad 12, hoja mínima 1,
`max_features=1.0`, semilla 42 y prueba cronológica de 20%. Cambia los
hiperparámetros directamente desde la CLI, sin editar el script:

```powershell
python run_experiment.py --n-estimators 160 --max-depth 8 --min-samples-leaf 2 --max-features 0.8 --random-state 42
```

`--data` permite indicar otro CSV y `--experiment-name` cambia el experimento.
La partición reserva siempre el tramo final ordenado por fecha y hora; las
imputaciones/codificación se ajustan solo con entrenamiento. Los predictores
numéricos se imputan con mediana y las categorías se codifican con One-Hot.

## Veinte corridas para comparar hiperparámetros

Con el entorno virtual activo, ejecutar desde esta carpeta:

```powershell
python run_experiments.py
```

El lanzador realiza un barrido **de un factor a la vez**: cinco valores de cada
uno de cuatro hiperparámetros (`n_estimators`, `max_depth`, `min_samples_leaf`
y `max_features`). En cada barrido los otros tres parámetros, la semilla y el
holdout temporal se mantienen fijos. Cada parámetro se registra junto con su
factor y valor bajo el experimento `SeoulBikeData - sensibilidad hiperparametros
(20 corridas)`. La configuración de referencia se repite una vez por barrido,
para poder medir el cambio de su factor con los otros constantes. El informe
compara el rango de MAE dentro de cada barrido como estimación práctica de
influencia; el diseño no mide interacciones entre hiperparámetros.

Para repetir una configuración manualmente, usa los mismos argumentos y la
misma semilla y contrasta sus métricas en MLflow.

Los resultados de estas 20 corridas, la configuración ganadora y el rango de
MAE por factor están resumidos en [ANALISIS.md](ANALISIS.md), con tabla e imagen
de sensibilidad exportadas desde los registros de MLflow.

## Declaración de uso de Copilot

Se utilizó Copilot únicamente como apoyo para revisar la redacción y la
ortografía de este README.
