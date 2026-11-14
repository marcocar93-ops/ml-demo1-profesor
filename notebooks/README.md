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
| 16 | [clase_16_profiling_greenai.ipynb](./clase_16_profiling_greenai.ipynb) | Profiling, comparación scikit-learn/ONNX y estimación didáctica de Green AI |

## Material de apoyo

Estos notebooks son ejercicios complementarios; no forman parte de la numeración principal:

- [ConvertirDB.ipynb](./ConvertirDB.ipynb): lectura y conversión entre CSV, Parquet, JSON y SQL.
- [Cargar parquet.ipynb](./Cargar%20parquet.ipynb): práctica breve de lectura y comparación de formatos.
- [PipelinePrueba.ipynb](./PipelinePrueba.ipynb): pipeline binario educativo con Breast Cancer Wisconsin (Diagnostic), desde la validación cruzada hasta la elección de umbral y la evaluación final. El dataset está incluido en scikit-learn y no requiere descarga; las métricas se guardan en [`reports/pipeline_prueba_wisconsin/`](../reports/pipeline_prueba_wisconsin/). No usar para diagnóstico ni decisiones clínicas.

La Clase 16 guarda sus mediciones y estimaciones en `reports/profiling_comparativo.csv` y `reports/green_ai.csv`, con gráficos en `reports/figures/`. Compara el RF final del Bloque 2 con una exportación ONNX e incluye el escalador ajustado en ambos artefactos. Las emisiones usan supuestos didácticos y no son una medición de energía; si ya existe `models/modelo.onnx`, el notebook conserva ese archivo y crea `models/modelo_clase16.onnx`.

## Datos y ejecución

La secuencia principal usa `data/raw/Crop_recommendation.csv` y reutiliza artefactos como `data/processed/crop.parquet` y los modelos del proyecto; verifica que los archivos requeridos existan antes de ejecutar cada clase. `PipelinePrueba.ipynb` es independiente de esos datos y modelos agrícolas. Ejecuta las celdas en orden dentro de cada notebook.

Las clases 1–6 trabajan la recomendación multiclase de cultivos. Desde la clase 7 se utiliza además una formulación binaria (`rice` / `no rice`) para estudiar factibilidad, métricas, costos y optimización; sus resultados no son directamente comparables con los de 22 clases. Las clases 11–12 ilustran aprendizaje por refuerzo en CartPole, un entorno de simulación distinto.
