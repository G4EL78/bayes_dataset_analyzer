import streamlit as st
import pandas as pd
import os
from src.bayes_model import cargar_datos, predecir_naive_bayes

# Configuración de la página
st.set_page_config(
    page_title="Analizador Teorema de Bayes - Películas",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Analizador Predictivo de Películas (Teorema de Bayes & Naive Bayes)")
st.markdown("""
Esta aplicación utiliza **probabilidad condicional y el Teorema de Bayes** implementados desde cero 
para predecir si una película obtendrá una **Calificación Alta (>8.0/10)** basándose en su género, presupuesto y tipo de lanzamiento.
""")

# Cargar datos (Apuntando correctamente a la carpeta data/raw/)
ruta_csv = "data/raw/peliculas.csv"
if not os.path.exists(ruta_csv):
    st.error(f"No se encontró el archivo de datos en '{ruta_csv}'. Revisa la estructura de carpetas.")
    st.stop()

df = cargar_datos(ruta_csv)

# Menú lateral
st.sidebar.header("Menú de Navegación")
opcion = st.sidebar.radio("Selecciona una sección:", [
    "1. Exploración del Dataset", 
    "2. Motor de Probabilidad Bayesiana", 
    "3. Predicción Interactiva (Naive Bayes)"
])

if opcion == "1. Exploración del Dataset":
    st.subheader("📊 Dataset de Películas (2016-2026)")
    st.write(f"Total de registros limpios en el análisis: **{len(df)}** películas.")
    st.dataframe(df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Distribución de Calificaciones Altas")
        conteo_cal = df["Calificacion_Alta"].value_counts()
        st.bar_chart(conteo_cal)
    with col2:
        st.markdown("### Películas por Género")
        conteo_gen = df["Genero"].value_counts()
        st.bar_chart(conteo_gen)

elif opcion == "2. Motor de Probabilidad Bayesiana":
    st.subheader("📐 Cálculos Base del Teorema de Bayes")
    st.markdown("Aquí se muestran las probabilidades a priori calculadas directamente del dataset:")
    
    total = len(df)
    cant_si = len(df[df["Calificacion_Alta"] == "Sí"])
    cant_no = len(df[df["Calificacion_Alta"] == "No"])
    
    p_si = cant_si / total
    p_no = cant_no / total
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric(label="Probabilidad P(Calificación Alta = Sí)", value=f"{p_si:.2f} ({cant_si}/{total})")
    with c2:
        st.metric(label="Probabilidad P(Calificación Alta = No)", value=f"{p_no:.2f} ({cant_no}/{total})")
        
    st.markdown("---")
    st.markdown("### Frecuencias por Género y Calificación")
    tabla_cruzada = pd.crosstab(df["Genero"], df["Calificacion_Alta"])
    st.dataframe(tabla_cruzada, use_container_width=True)

elif opcion == "3. Predicción Interactiva (Naive Bayes)":
    st.subheader("🔮 Simulador de Clasificación Naive Bayes")
    st.markdown("Selecciona las características de una película para calcular la probabilidad de éxito crítico:")
    
    with st.form("form_prediccion"):
        genero_sel = st.selectbox("Género:", df["Genero"].unique())
        presupuesto_sel = st.selectbox("Presupuesto:", df["Presupuesto"].unique())
        lanzamiento_sel = st.selectbox("Tipo de Lanzamiento:", df["Tipo_Lanzamiento"].unique())
        
        boton_calcular = st.form_submit_button("Calcular Probabilidad Bayesiana")
        
    if boton_calcular:
        caracteristicas = {
            "Genero": genero_sel,
            "Presupuesto": presupuesto_sel,
            "Tipo_Lanzamiento": lanzamiento_sel
        }
        
        clase_predicha, probabilidad, detalles = predecir_naive_bayes(df, caracteristicas)
        
        st.markdown("---")
        st.subheader("🎯 Resultado del Modelo")
        
        if clase_predicha == "Sí":
            st.success(f"**Predicción:** ¡Calificación Alta (>8.0)! (Probabilidad: {probabilidad * 100:.2f}%)")
        else:
            st.error(f"**Predicción:** Calificación Baja o Moderada (Probabilidad de no ser alta: {probabilidad * 100:.2f}%)")
            
        st.markdown("### Desglose matemático por variable (Verosimilitud):")
        st.json(detalles)
