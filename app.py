"""
App web de inferencia para predicción de cultivos.
Asignatura: Machine Learning (RIA-805)
Bloque 1 - Cierre: Modelo entrenado en producción local.

Ejecutar con: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "notebooks" / "models"
FEATURE_COLUMNS = [
    "N", "P", "K", "temperature", "humidity", "ph", "rainfall"
]

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Predicción de Cultivos - RIA-805",
    page_icon="🌱",
    layout="wide"
)

# ============================================================
# CARGAR MODELO, SCALER Y LABEL ENCODER
# ============================================================
@st.cache_resource
def cargar_modelos():
    """Carga el modelo, scaler y label encoder desde disco."""
    modelo = joblib.load(MODEL_DIR / "random_forest.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    le = joblib.load(MODEL_DIR / "label_encoder.joblib")

    if list(modelo.feature_names_in_) != FEATURE_COLUMNS:
        raise ValueError("Las features del modelo no coinciden con las esperadas por la app.")
    if list(scaler.feature_names_in_) != FEATURE_COLUMNS:
        raise ValueError("Las features del scaler no coinciden con las esperadas por la app.")
    if not np.array_equal(modelo.classes_, np.arange(len(le.classes_))):
        raise ValueError("Las clases del modelo no coinciden con el label encoder.")

    return modelo, scaler, le

try:
    modelo, scaler, le = cargar_modelos()
    modelos_cargados = True
except Exception as e:
    modelos_cargados = False
    st.error(f"Error al cargar modelos: {e}")
    st.info("Asegúrate de haber ejecutado el pipeline de entrenamiento primero.")


@st.cache_data
def cargar_datos_evaluacion():
    """Carga las features y etiquetas usadas para evaluar el modelo."""
    ruta_datos = ROOT_DIR / "data" / "processed" / "crop.parquet"
    if not ruta_datos.is_file():
        raise FileNotFoundError(
            f"No se encontró el dataset de evaluación: {ruta_datos}"
        )

    datos = pd.read_parquet(ruta_datos)
    columnas_faltantes = set(FEATURE_COLUMNS + ["label"]) - set(datos.columns)
    if columnas_faltantes:
        raise ValueError(
            "Faltan columnas requeridas en el dataset: "
            + ", ".join(sorted(columnas_faltantes))
        )

    return datos[FEATURE_COLUMNS], datos["label"]


def calcular_costos_multiclase(
    etiquetas_reales,
    predicciones,
    etiqueta_rechazo,
    costo_fp,
    costo_fn
):
    """Calcula FP, FN y costo total con rechazo de predicciones de baja confianza."""
    rechazadas = predicciones == etiqueta_rechazo
    errores = predicciones != etiquetas_reales

    # Una predicción errónea aceptada suma un FP y un FN; un rechazo, solo un FN.
    falsos_positivos = int(np.count_nonzero(errores & ~rechazadas))
    falsos_negativos = int(np.count_nonzero(errores))
    costo_total = falsos_positivos * costo_fp + falsos_negativos * costo_fn
    return falsos_positivos, falsos_negativos, costo_total


# ============================================================
# TÍTULO Y DESCRIPCIÓN
# ============================================================
st.title("🌱 Sistema de Predicción de Cultivos")
st.markdown("""
Esta aplicación usa un modelo de **Random Forest** entrenado con el dataset
**Crop Recommendation** para recomendar el cultivo más adecuado según las
condiciones del suelo y el clima.

**Asignatura:** Machine Learning (RIA-805)
**Carrera:** Ingeniería en Robótica e Inteligencia Artificial
**Dataset:** Crop Recommendation (2,200 muestras, 7 features, 22 cultivos)
""")

if not modelos_cargados:
    st.stop()

# ============================================================
# SIDEBAR: INFORMACIÓN DEL MODELO
# ============================================================
st.sidebar.header("📊 Información del modelo")
st.sidebar.markdown(f"""
- **Modelo:** Random Forest
- **Features:** 7
- **Clases:** {len(le.classes_)}
- **Dataset:** Crop Recommendation
""")

st.sidebar.header("🌾 Cultivos disponibles")
for i, cultivo in enumerate(le.classes_, 1):
    st.sidebar.markdown(f"{i}. {cultivo}")

tab_prediccion, tab_factibilidad = st.tabs(
    ["Predicción", "Factibilidad Económica"]
)

# ============================================================
# FORMULARIO DE ENTRADA
# ============================================================
with tab_prediccion:
    st.header("📝 Ingresa los datos del terreno")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Nutrientes del suelo")
        N = st.number_input("Nitrógeno (N)", min_value=0.0, max_value=200.0, value=90.0, step=1.0)
        P = st.number_input("Fósforo (P)", min_value=0.0, max_value=200.0, value=42.0, step=1.0)
        K = st.number_input("Potasio (K)", min_value=0.0, max_value=300.0, value=43.0, step=1.0)

    with col2:
        st.subheader("Condiciones climáticas")
        temperature = st.number_input("Temperatura (°C)", min_value=0.0, max_value=50.0, value=20.8, step=0.1)
        humidity = st.number_input("Humedad (%)", min_value=0.0, max_value=100.0, value=82.0, step=0.1)
        ph = st.number_input("pH del suelo", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        rainfall = st.number_input("Precipitación (mm)", min_value=0.0, max_value=500.0, value=202.9, step=0.1)

    # ============================================================
    # BOTÓN DE PREDICCIÓN
    # ============================================================
    if st.button("🔮 Predecir cultivo", type="primary", use_container_width=True):
        # Crear dataframe con los datos de entrada
        datos_entrada = pd.DataFrame({
            'N': [N],
            'P': [P],
            'K': [K],
            'temperature': [temperature],
            'humidity': [humidity],
            'ph': [ph],
            'rainfall': [rainfall]
        })

        # Escalar
        datos_escalados = scaler.transform(datos_entrada)

        # Predecir
        prediccion = modelo.predict(datos_escalados)[0]
        probabilidades = modelo.predict_proba(datos_escalados)[0]

        # Obtener nombre del cultivo
        cultivo_predicho = le.inverse_transform([prediccion])[0]
        confianza = probabilidades[prediccion] * 100

        # ============================================================
        # MOSTRAR RESULTADO
        # ============================================================
        st.success(f"### 🌾 Cultivo recomendado: **{cultivo_predicho.upper()}**")
        st.metric("Confianza del modelo", f"{confianza:.2f}%")

        # Top 5 predicciones
        st.subheader("📊 Top 5 cultivos más probables")
        top_5_idx = np.argsort(probabilidades)[::-1][:5]
        top_5_cultivos = le.inverse_transform(top_5_idx)
        top_5_probs = probabilidades[top_5_idx]

        df_top5 = pd.DataFrame({
            'Cultivo': top_5_cultivos,
            'Probabilidad (%)': (top_5_probs * 100).round(2)
        })
        st.dataframe(df_top5, width="stretch")

        # Gráfico de barras
        st.bar_chart(df_top5.set_index('Cultivo')['Probabilidad (%)'])

        # ============================================================
        # INTERPRETACIÓN
        # ============================================================
        if confianza >= 80:
            st.success("✅ El modelo tiene alta confianza en esta predicción.")
        elif confianza >= 50:
            st.warning("⚠️ El modelo tiene confianza moderada. Considera recolectar más datos.")
        else:
            st.error("❌ El modelo tiene baja confianza. Los datos podrían estar fuera del dominio de entrenamiento.")

with tab_factibilidad:
    st.header("💰 Análisis de factibilidad económica")
    st.markdown(
        "Ajusta el costo de cada tipo de error para encontrar el umbral de "
        "confianza que minimiza el costo total en el conjunto de prueba."
    )

    if not hasattr(modelo, "predict_proba"):
        st.error("El modelo cargado no permite calcular probabilidades por clase.")
    else:
        col_cfp, col_cfn = st.columns(2)
        with col_cfp:
            costo_fp = st.slider(
                "Costo por falso positivo (C_FP)",
                min_value=1,
                max_value=10_000,
                value=100,
                step=1,
                format="$%d",
            )
        with col_cfn:
            costo_fn = st.slider(
                "Costo por falso negativo (C_FN)",
                min_value=1,
                max_value=10_000,
                value=100,
                step=1,
                format="$%d",
            )

        X, y = cargar_datos_evaluacion()
        y_codificado = le.transform(y)
        X_escalado = pd.DataFrame(
            scaler.transform(X), columns=X.columns, index=X.index
        )
        _, X_prueba, _, y_prueba = train_test_split(
            X_escalado,
            y_codificado,
            test_size=0.2,
            random_state=42,
            stratify=y_codificado,
        )
        probabilidades = modelo.predict_proba(X_prueba)
        clases_predichas = modelo.classes_[np.argmax(probabilidades, axis=1)]
        etiquetas_predichas = le.inverse_transform(clases_predichas)
        etiquetas_reales = le.inverse_transform(y_prueba)
        confianza = np.max(probabilidades, axis=1)
        etiqueta_rechazo = "Sin recomendación"
        umbrales = np.round(np.arange(0.1, 1.0, 0.1), 1)

        # Bajo el umbral se abstiene de recomendar; el umbral 0.5 se evalúa igual que los demás.
        predicciones_05 = np.where(
            confianza >= 0.5, etiquetas_predichas, etiqueta_rechazo
        )
        etiquetas_matriz = list(le.classes_) + [etiqueta_rechazo]
        matriz = confusion_matrix(
            etiquetas_reales,
            predicciones_05,
            labels=etiquetas_matriz,
        )
        matriz_df = pd.DataFrame(
            matriz,
            index=pd.Index(etiquetas_matriz, name="Real"),
            columns=pd.Index(etiquetas_matriz, name="Predicción"),
        )

        costos = []
        for umbral in umbrales:
            predicciones = np.where(
                confianza >= umbral, etiquetas_predichas, etiqueta_rechazo
            )
            _, _, costo_total = calcular_costos_multiclase(
                etiquetas_reales,
                predicciones,
                etiqueta_rechazo,
                costo_fp,
                costo_fn,
            )
            costos.append(costo_total)

        indice_optimo = int(np.argmin(costos))
        umbral_optimo = float(umbrales[indice_optimo])
        costo_optimo = int(costos[indice_optimo])
        _, _, costo_05 = calcular_costos_multiclase(
            etiquetas_reales,
            predicciones_05,
            etiqueta_rechazo,
            costo_fp,
            costo_fn,
        )
        ahorro = costo_05 - costo_optimo

        st.subheader("Matriz de confusión con umbral 0.5")
        st.caption(
            "Las filas son cultivos reales y las columnas son predicciones; "
            "“Sin recomendación” identifica casos por debajo del umbral."
        )
        st.dataframe(matriz_df, width="stretch", height=500)

        st.subheader("Costo total según el umbral")
        datos_costos = pd.DataFrame(
            {"Umbral": umbrales, "Costo total ($)": costos}
        )
        st.line_chart(
            datos_costos,
            x="Umbral",
            y="Costo total ($)",
            alt="Curva del costo total para umbrales de confianza de 0.1 a 0.9",
        )
        st.success(
            f"Umbral óptimo: **{umbral_optimo:.1f}** "
            f"(costo mínimo: **${costo_optimo:,.2f}**)."
        )

        col_costo_05, col_costo_optimo, col_ahorro = st.columns(3)
        col_costo_05.metric("Costo con umbral 0.5", f"${costo_05:,.2f}")
        col_costo_optimo.metric("Costo con umbral óptimo", f"${costo_optimo:,.2f}")
        col_ahorro.metric(
            "Ahorro potencial" if ahorro >= 0 else "Aumento de costo",
            f"${abs(ahorro):,.2f}",
        )
        st.caption(
            "El costo usa el enfoque multiclase uno-contra-el-resto: una "
            "clasificación incorrecta aceptada cuenta como FP y FN; un rechazo "
            "cuenta como FN de la clase real."
        )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("RIA-805 Machine Learning — Universidad Católica de Cuenca — 2025")
