# Secuencia de clases

Los notebooks `clase_1` a `clase_9` forman la secuencia principal del curso. Los nombres y números reflejan la progresión desde comprender los datos hasta evaluar decisiones y costos.

| Clase | Notebook | Tema |
|---|---|---|
| 1 | [clase_1_contexto_datos.ipynb](./clase_1_contexto_datos.ipynb) | Contexto del problema, variables y límites del dataset |
| 2 | [clase_2_analisis_exploratorio.ipynb](./clase_2_analisis_exploratorio.ipynb) | Calidad, distribuciones y relaciones entre variables |
| 3 | [clase_3_preprocesamiento.ipynb](./clase_3_preprocesamiento.ipynb) | Imputación, escalado y prevención de data leakage |
| 4 | [clase_4_aprendizaje_supervisado.ipynb](./clase_4_aprendizaje_supervisado.ipynb) | Entrenamiento, evaluación e interpretación de clasificadores |
| 5 | [clase_5_no_supervisado.ipynb](./clase_5_no_supervisado.ipynb) | K-Means, DBSCAN y reducción dimensional |
| 6 | [clase_6_integracion.ipynb](./clase_6_integracion.ipynb) | Integración del pipeline, benchmarking y buenas prácticas |
| 7 | [clase_7_factibilidad.ipynb](./clase_7_factibilidad.ipynb) | Factibilidad técnica y pertinencia |
| 8 | [clase_8_metricas.ipynb](./clase_8_metricas.ipynb) | Métricas para escenarios desbalanceados |
| 9 | [clase_9_costos_threshold.ipynb](./clase_9_costos_threshold.ipynb) | Matrices de costo y ajuste del umbral |

## Material de apoyo

Estos notebooks son ejercicios complementarios; no forman parte de la numeración principal:

- [ConvertirDB.ipynb](./ConvertirDB.ipynb): lectura y conversión entre CSV, Parquet, JSON y SQL.
- [Cargar parquet.ipynb](./Cargar%20parquet.ipynb): práctica breve de lectura y comparación de formatos.

## Datos y ejecución

La secuencia usa `data/raw/Crop_recommendation.csv` para la comprensión inicial y el análisis exploratorio. Las clases posteriores reutilizan artefactos como `data/processed/crop.parquet` y modelos guardados en `models/`; verifica que esos archivos existan antes de ejecutar las celdas que los requieren. Ejecuta las celdas en orden dentro de cada notebook.
