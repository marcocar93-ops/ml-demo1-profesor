# Machine Learning - Práctica 1: Pipeline End-to-End para Robótica

**Estudiante:** 
**Asignatura:** Machine Learning (8.º Ciclo - Ingeniería en Robótica)  
**Dataset:** Wall-Following Robot Navigation Data (UCI)

## Descripción del Proyecto
Este proyecto implementa un pipeline completo de Machine Learning modular, reproducible e industrial sobre datos de un robot móvil SCITOS-G5 equipado con 24 sensores ultrasónicos.

## Estructura del Repositorio
.
├── data/
│   ├── raw/          # Datos crudos (UCI)
│   └── processed/    # Datos procesados (.parquet, JSONs)
├── models/           # Modelos entrenados (.pkl)
├── notebooks/        # Análisis exploratorio y pruebas (.ipynb)
├── src/              # Código modular en Python (.py)
├── README.md         # Documentación del proyecto
└── requirements.txt  # Dependencias del proyecto

## Requisitos e Instalación
```bash
python -m venv venv
source venv/bin/activate  # En Windows: .\venv\Scripts\Activate
pip install -r requirements.txt
```

## Ejecutar la app web

```bash
pip install -r requirements_app.txt
streamlit run app.py
```

La app se abrirá en `http://localhost:8501`.

La app utiliza el Random Forest multiclase y los artefactos de preprocesamiento
guardados en `notebooks/models/`. Antes de abrirla, ejecuta la celda de
entrenamiento de `notebooks/clase_6_integracion.ipynb`; esa celda guarda el
modelo, el `RobustScaler` y el `LabelEncoder` juntos para evitar mezclar
artefactos de entrenamientos distintos.

### Factibilidad económica

En la pestaña **Factibilidad Económica** puedes asignar costos de falsos
positivos y falsos negativos, consultar la matriz de confusión con umbral
0.5 y comparar el costo total para umbrales de confianza de 0.1 a 0.9. La
evaluación usa un conjunto de prueba estratificado (20 %, semilla 42).

Como el modelo recomienda entre varios cultivos, las predicciones con
confianza menor al umbral se registran como **Sin recomendación**. Una
clasificación incorrecta aceptada cuenta como un falso positivo para el cultivo
predicho y un falso negativo para el cultivo real; un rechazo cuenta como un
falso negativo.
