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
import os

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
    modelo = joblib.load('models/random_forest.pkl')
    scaler = joblib.load('models/scaler.pkl')
    le = joblib.load('models/label_encoder.pkl')
    return modelo, scaler, le

try:
    modelo, scaler, le = cargar_modelos()
    modelos_cargados = True
except Exception as e:
    modelos_cargados = False
    st.error(f"Error al cargar modelos: {e}")
    st.info("Asegúrate de haber ejecutado el pipeline de entrenamiento primero.")

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

# ============================================================
# FORMULARIO DE ENTRADA
# ============================================================
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
    st.dataframe(df_top5, use_container_width=True)
    
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

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("RIA-805 Machine Learning — Universidad Católica de Cuenca — 2025")
