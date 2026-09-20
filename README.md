# Gas Classification under Sensor Drift

Clasificación multiclase de gases a partir de un array de sensores químicos, con especial atención a la **generalización entre periodos de adquisición**.

El objetivo del ejercicio no es únicamente obtener un buen clasificador, sino comprobar hasta qué punto una evaluación aleatoria convencional representa el comportamiento esperado cuando las respuestas de los sensores cambian entre campañas de adquisición.

## Dataset

Se utiliza **Gas Sensor Array Drift Dataset** de UCI (ID 224, DOI `10.24432/C5RP6W`), con:

- 13.910 observaciones
- 128 features
- 6 gases
- 10 batches de adquisición
- 16 sensores × 8 descriptores por sensor

Los ficheros originales se conservan en `data/raw/`. `scripts/prepare_data.py` transforma el formato original a una tabla con `label`, `batch` y `feature_1...feature_128`. No aplica escalado, imputación, PCA ni eliminación de outliers: cualquier transformación aprendida se ajusta después exclusivamente sobre el training de cada split.

El dataset no presenta valores ausentes, infinitos ni duplicados exactos, pero sí grandes diferencias de escala y una redundancia considerable entre features.

## Por qué este dataset

La búsqueda se centró finalmente en datasets de **arrays de sensores químicos**, porque permitían plantear un problema industrial claro de clasificación bajo cambios en la respuesta instrumental.

Se consideraron varias alternativas cercanas:

- Gas Sensor Array Drift at Different Concentrations: Aporta una extensión muy cercana que incorpora concentración. No se eligió como dataset principal porque,   aunque resulta interesante para separar concentración y drift, el objetivo era medir la robustez de la clasificación sin asumir que la concentración estuviera disponible en inferencia; se conserva como primera extensión futura.
- Gas sensor array under dynamic gas mixtures: Aporta señales temporales continuas de 16 sensores frente a mezclas dinámicas. Se descartó como principal porque su enfoque se orienta a mezclas, concentraciones y señales continuas, en lugar de a la generalización entre campañas separadas por drift.
- Gas sensor array under flow modulation: Aporta señales crudas y features bajo mezclas de acetona y etanol. No se seleccionó como principal debido a que cuenta con muy pocas series independientes y su foco está en la modulación de flujo y concentración, no en la degradación a largo plazo.
- Gas sensor array exposed to turbulent gas mixtures: Aporta un entorno de plumas turbulentas más realista. Quedó fuera como dataset principal porque prioriza la respuesta a la turbulencia y mezclas espaciales, sin ofrecer una estructura tan directa para evaluar el drift entre periodos.

También se exploraron otros dominios industriales y ambientales (mantenimiento hidráulico, defectos en acero, consumo energético industrial o calidad del aire), pero se descartaron cuando la pregunta experimental resultaba menos clara o demasiado trivial.

El dataset elegido ofrece algo especialmente útil para esta prueba: la variable `batch` permite construir una evaluación temporal explícita y contrastarla con random CV. Así, el interés no está sólo en “clasificar seis gases”, sino en estudiar **si el modelo sigue funcionando cuando cambian las condiciones de adquisición**.

## Limitaciones

### Condiciones experimentales controladas

Las medidas proceden de exposiciones controladas a gases. Un entorno industrial real puede incluir mezclas, humedad, interferencias o condiciones operativas distintas.

Por ello, los resultados no deben interpretarse como rendimiento de un detector listo para producción, sino como evidencia de robustez dentro de este sistema experimental.

### Composición desigual de los batches

Los batches cambian mucho en tamaño y prevalencia de clases; algunas clases incluso están ausentes en determinados periodos.

Esto puede confundir dos fenómenos distintos: cambios en las proporciones de clases (`P(Y)`) y cambios en la respuesta de los sensores para una misma clase (`P(X|Y)`). Por ese motivo el EDA incluye también análisis condicionados por clase y los resultados temporales se inspeccionan batch a batch.

### Concentración no disponible por observación

La versión utilizada no incluye la concentración como columna por muestra. Por tanto, no puede atribuirse todo cambio entre batches exclusivamente a sensor drift: parte podría deberse a diferencias en concentración u otras condiciones experimentales.

### Features preextraídas

Las 128 variables son descriptores de las respuestas de los sensores, no las señales temporales crudas.

Esto hace el problema reproducible y manejable, pero impide estudiar si una representación diferente de la señal podría ser más robusta al drift.

### Un único sistema experimental

El último batch temporal (B10) mide generalización hacia un periodo posterior del mismo sistema, no hacia otro laboratorio, sensor array o instalación. La validez externa sigue por tanto sin comprobarse.

## Protocolo de evaluación

La evaluación separa claramente **selección de modelo** y **test final**.


### Desarrollo: Batch 1 (B1) - Batch 9 (B9)

Todos los modelos e hiperparámetros se comparan únicamente con B1-B9 mediante dos protocolos complementarios.

**Random stratified 5-fold CV** se mantiene como referencia IID-like. Mezcla muestras de distintos batches entre train y validation, por lo que responde a la pregunta: “¿qué ocurre si entrenamiento y validación contienen condiciones de adquisición similares?”.

**Expanding-window temporal validation** entrena siempre con batches anteriores y valida sobre el siguiente:

```text
B1      -> B2
B1-B2   -> B3
...
B1-B8   -> B9
```

Este protocolo es el criterio principal de selección porque se aproxima mejor al uso real que queremos estudiar: entrenar con datos históricos y predecir un periodo futuro cuyas condiciones aún no se han observado.

`batch` se usa únicamente para construir estos splits y analizar resultados; nunca entra como feature del modelo, evitando que el clasificador explote un identificador de dominio que no estaría disponible de forma útil para generalizar a un batch nuevo.

### Test final: B10

B10 permanece completamente aislado durante el desarrollo. Sólo después de fijar modelo e hiperparámetros se reentrena el pipeline seleccionado sobre B1-B9 y se evalúa una única vez sobre B10.

La separación evita seguir adaptando decisiones al mismo conjunto que se utiliza para reportar el resultado final.

## Métrica

La métrica principal es **macro-F1**, porque la distribución de gases varía considerablemente entre batches y queremos que cada clase tenga el mismo peso en la evaluación.

También se reportan accuracy, F1 por clase y matrices de confusión para evaluar el comportamiento predictivo específico de cada clase. Cuando una clase no aparece en un batch de validación, su F1 se mantiene como `NaN` en lugar de asignarle artificialmente cero.

## Modelos

### Logistic Regression

Se utiliza como baseline lineal:

```text
StandardScaler -> LogisticRegression
```

El escalado es importante porque las features presentan órdenes de magnitud muy distintos. Al implementarse mediante `Pipeline`, el scaler se ajusta únicamente sobre cada conjunto de entrenamiento.

Se estudia `C ∈ {0.01, 0.1, 1, 10}`. La mejor configuración temporal es `C=10`.

### XGBoost

Se utiliza como contraste no lineal para comprobar si una mayor capacidad para modelar interacciones entre s  logreg_tuning.csv
  xgboost_tuning.csv
  final_evaluation.csvensores mejora la generalización bajo shift.

Se mantiene el resto de parámetros fijo y se compara `max_depth ∈ {2, 4, 6}`. La mejor variante temporal es `max_depth=4`.

No se realiza una búsqueda exhaustiva: el objetivo es comparar hipótesis de modelado, no optimizar agresivamente sobre sólo ocho periodos de validación.

## Resultados

| Modelo | Random CV macro-F1 | Temporal macro-F1 medio |
| --- | ---: | ---: |
| Logistic Regression (`C=10`) | ~0.991 | **~0.813** |
| XGBoost (`max_depth=4`) | ~0.993 | ~0.747 |

Random CV presenta ambos modelos como prácticamente perfectos. Sin embargo, al validar sobre batches futuros el rendimiento cae de forma importante, y Logistic Regression supera a XGBoost en la mayoría de periodos.

La conclusión principal es que **más capacidad predictiva bajo una partición aleatoria no implica mayor robustez frente a batch-dependent distribution shift**.

Las matrices de confusión muestran además que la degradación no es uniforme: determinadas clases concentran buena parte de los errores y ese patrón cambia entre batches.

### Holdout final

Tras seleccionar `Logistic Regression (C=10)` usando exclusivamente B1-B9, se entrena sobre todo el desarrollo y se evalúa una sola vez en B10:

| Métrica | Resultado |
| --- | ---: |
| Macro-F1 | **0.6632** |
| Accuracy | **0.6750** |
| F1 clase 1 | 0.6438 |
| F1 clase 2 | 0.8826 |
| F1 clase 3 | 0.5262 |
| F1 clase 4 | 0.4792 |
| F1 clase 5 | 0.7229 |
| F1 clase 6 | 0.7246 |

B10 queda cerca de los peores periodos observados durante desarrollo. Esto refuerza la conclusión de que el problema central no era optimizar unas décimas en random CV, sino generalizar frente a cambios entre campañas.

Después de observar B10 no se modifica el modelo, los hiperparámetros, las features ni el preprocessing.

## Estructura

```text
data/                               # Datos utilizados por el proyecto
  raw/                              # Ficheros originales del dataset UCI, sin modificar
  processed/                        # Dataset tabular generado por prepare_data.py; gitignored

notebooks/                          # Análisis exploratorio e interpretación de resultados
  01_eda.ipynb                      # Calidad de datos, distribución por batches, drift y PCA
  02_model_evaluation.ipynb         # Tuning, comparación de protocolos, errores por clase y test final

results/                            # CSVs versionados con los resultados de los experimentos

scripts/                            # Puntos de entrada ejecutables desde línea de comandos
  prepare_data.py                   # Convierte los .dat originales al formato tabular procesado
  inspect_splits.py                 # Muestra y valida la composición de los splits de evaluación
  run_experiments.py                # Ejecuta los experimentos definidos y guarda sus resultados

src/                                # Lógica reutilizable del pipeline experimental
  config.py                         # Rutas y configuración común del proyecto
  evaluation.py                     # Definición de splits y métricas de clasificación
  experiments.py                    # Ejecución genérica de modelos sobre protocolos de evaluación
  models.py                         # Definiciones de Logistic Regression y XGBoost

requirements.txt                    # Dependencias necesarias para reproducir el proyecto
```

Los CSV de `results/` se versionan para poder revisar el análisis sin reentrenar los modelos.

## Instalación y ejecución

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m scripts.prepare_data
```

Experimentos:

```bash
# Ajuste de la regularización de Logistic Regression usando B1-B9
python -m scripts.run_experiments --experiment logreg_tuning

# Comparación de distintas profundidades de XGBoost usando los mismos protocolos de desarrollo
python -m scripts.run_experiments --experiment xgboost_tuning

# Entrenamiento del modelo seleccionado sobre B1-B9 y evaluación única sobre el holdout B10
python -m scripts.run_experiments --experiment final_evaluation
```

Para inspeccionar los splits:

```bash
python -m scripts.inspect_splits
```

El análisis completo está en:

```text
notebooks/01_eda.ipynb
notebooks/02_model_evaluation.ipynb
```

## Trabajo futuro

- **Incorporar concentración como metadato de análisis.** Permitiría separar mejor drift instrumental de cambios debidos a concentración y comprobar si el rendimiento cae de forma distinta según el rango de exposición.
- **Evaluar adaptación explícita al drift.** Si en producción llegan nuevas etiquetas, métodos online/adaptativos podrían actualizar el modelo en lugar de mantenerlo fijo durante toda su vida útil.
- **Estudiar domain adaptation.** Si fuese posible observar datos no etiquetados del siguiente periodo antes de inferir, podrían aprovecharse para reducir el cambio entre dominios sin requerir labels.
- **Analizar selección de sensores/features.** El EDA muestra que algunos sensores cambian más que otros. Estudiar subconjuntos podría mejorar robustez y, además, reducir coste de instrumentación.
- **Trabajar sobre señal cruda y validar externamente.** Las señales temporales permitirían diseñar representaciones potencialmente más invariantes al drift, mientras que otro sensor array o instalación permitiría medir generalización real fuera de este sistema.

## Uso de IA

Se utilizaron asistentes de IA para acelerar tareas de implementación y ayudar a estructurar el formato Markdown de parte de la documentación.

**Decisiones asumidas y defendidas personalmente:** búsqueda y selección del dataset y del problema, diseño del protocolo temporal de evaluación, reserva de B10 para test, elección de métricas de evaluación, selección de modelos e hiperparámetros, interpretación de resultados y decisión de no modificar el pipeline después de observar el test.

**Partes desarrolladas con asistencia:** generación de parte del código, refactorizaciones y generación de celdas repetitivas de análisis.

Todos los resultados reportados proceden de la ejecución del código versionado en el repositorio.

