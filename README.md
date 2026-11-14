# Machine Learning: ruta didáctica de datos a producción

Repositorio de notebooks para enseñar habilidades de Machine Learning de forma progresiva: comprensión y preparación de datos, aprendizaje supervisado y no supervisado, evaluación, costos, optimización, aprendizaje por refuerzo e introducción a MLOps.

La secuencia principal usa el conjunto **Crop Recommendation** para practicar el flujo de datos y modelos. Como caso complementario, `PipelinePrueba.ipynb` aplica el mismo razonamiento a **Breast Cancer Wisconsin (Diagnostic)**, un conjunto médico incluido en scikit-learn y ejecutable sin descargar datos. Este segundo caso es únicamente educativo y no sirve para diagnóstico ni decisiones clínicas.

## Estructura del Repositorio
.
├── data/
│   ├── raw/          # Datos de entrada
│   └── processed/    # Datos preparados para las clases
├── models/           # Artefactos de modelos
├── notebooks/        # Clases y ejercicios complementarios
├── reports/          # Resultados reproducibles
├── app/              # API de inferencia
├── README.md         # Guía del proyecto
└── requirements*.txt # Dependencias por componente

## Requisitos e Instalación

### Configurar el entorno en Windows y VS Code

Haz estos pasos desde la carpeta raíz del proyecto (`ml-demo1-profesor`):

1. En VS Code, abre una terminal nueva con **Terminal > New Terminal**.
2. Si PowerShell bloquea la activación del entorno, permite scripts solo en esa terminal:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```

3. Activa el entorno virtual del proyecto:

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

   Al activarse, la terminal mostrará `(venv)` al inicio de la línea. Si ya está activo en esa terminal, no hace falta repetirlo; al abrir otra terminal, actívalo de nuevo.

4. Instala las dependencias del proyecto en ese entorno (solo es necesario hacerlo cuando configures el entorno o cambien las dependencias):

   ```powershell
   python -m pip install -r requirements.txt
   ```

   Para ejecutar el notebook de aprendizaje por refuerzo de la Clase 12, instala además:

   ```powershell
   python -m pip install stable-baselines3
   ```

5. En VS Code, abre **Ctrl+Shift+P**, ejecuta **Python: Select Interpreter** y selecciona el intérprete que termina en `venv\Scripts\python.exe` (no el de `.venv`).
6. En cada notebook, pulsa **Select Kernel** en la esquina superior derecha y elige el kernel de `venv`. Después de cambiarlo, reinicia el kernel y vuelve a ejecutar las celdas.
7. Para comprobar qué Python está usando la terminal o el notebook, ejecuta:

   ```python
   import sys
   print(sys.executable)
   ```

   La ruta debe terminar en `venv\Scripts\python.exe`. La activación de la terminal y la selección del kernel del notebook son independientes.

Si prefieres no activar el entorno en PowerShell, puedes instalar o ejecutar Python directamente con `.\venv\Scripts\python.exe -m pip ...` o `.\venv\Scripts\python.exe tu_script.py`.

En macOS/Linux, usa `source venv/bin/activate` en lugar del comando de activación de PowerShell.

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
