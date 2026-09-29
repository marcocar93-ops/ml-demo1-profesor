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
