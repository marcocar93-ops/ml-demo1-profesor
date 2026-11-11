# Secuencia de clases

Los notebooks `clase_1` a `clase_13` forman la secuencia principal hasta el cierre del Bloque 2. La Clase 14 sobre relevancia y sostenibilidad se trabaja en AA 3.2; la secuencia continúa con MLOps en la Clase 15.

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
| 10 | [clase_10_optuna.ipynb](./clase_10_optuna.ipynb) | Validación cruzada y optimización bayesiana con Optuna |
| 11 | [clase_11_rl_gymnasium.ipynb](./clase_11_rl_gymnasium.ipynb) | Introducción al aprendizaje por refuerzo con Gymnasium |
| 12 | [clase_12_rl_entrenamiento.ipynb](./clase_12_rl_entrenamiento.ipynb) | Entrenamiento y evaluación de un agente de aprendizaje por refuerzo |
| 13 | [clase_13_cierre_b2.ipynb](./clase_13_cierre_b2.ipynb) | Optimización de rendimiento y cierre del Bloque 2 |
| 14 | — | Relevancia y sostenibilidad (AA 3.2) |
| 15 | [clase_15_mlops_docker_fastapi.ipynb](./clase_15_mlops_docker_fastapi.ipynb) | MLOps, ONNX, FastAPI, Docker y profiling |

## Material de apoyo

Estos notebooks son ejercicios complementarios; no forman parte de la numeración principal:

- [ConvertirDB.ipynb](./ConvertirDB.ipynb): lectura y conversión entre CSV, Parquet, JSON y SQL.
- [Cargar parquet.ipynb](./Cargar%20parquet.ipynb): práctica breve de lectura y comparación de formatos.
- [PipelinePrueba.ipynb](./PipelinePrueba.ipynb): reproducción educativa del protocolo y la métrica del PhysioNet/CinC Challenge 2019 con predicción temprana de sepsis.

## Datos y ejecución

La secuencia usa `data/raw/Crop_recommendation.csv` para la comprensión inicial y el análisis exploratorio. Las clases posteriores reutilizan artefactos como `data/processed/crop.parquet` y modelos guardados en `models/`; verifica que esos archivos existan antes de ejecutar las celdas que los requieren. `PipelinePrueba.ipynb` descarga los archivos públicos del PhysioNet/CinC Challenge 2019 a `data/raw/physionet_challenge_2019/` cuando no están disponibles; esa carpeta de datos descargados está excluida de Git. Ejecuta las celdas en orden dentro de cada notebook.
