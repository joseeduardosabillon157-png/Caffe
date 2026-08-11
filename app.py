import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from groq import Groq

st.set_page_config(
    page_title="AGRODETECT - Detección Foliar Café",
    page_icon="🍃",
    layout="wide"
)

st.markdown("""
<style>
    /* Fondo principal y fuente */
    .main {
        background-color: #F8F6F0;
        color: #2D2B2A;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Contenedores de tarjetas */
    .card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    /* Encabezados y títulos */
    .header-title {
        font-size: 26px;
        font-weight: 700;
        color: #1A1918;
        margin-bottom: 5px;
    }
    .header-sub {
        font-size: 13px;
        color: #7A7875;
        margin-bottom: 20px;
    }
    
    /* Porcentaje de confianza */
    .confidence-score {
        font-size: 38px;
        font-weight: 800;
        color: #1A1918;
        text-align: right;
    }
    .confidence-label {
        font-size: 10px;
        font-weight: 700;
        color: #8C8A85;
        text-align: right;
        letter-spacing: 1px;
    }
    
    /* Secciones del reporte de Groq */
    .section-num {
        background-color: #1A1918;
        color: #FFFFFF;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
        margin-right: 8px;
    }
    .section-title {
        font-size: 14px;
        font-weight: 700;
        color: #2D2B2A;
    }
    .section-text {
        font-size: 13px;
        color: #4A4845;
        margin-top: 5px;
        margin-bottom: 18px;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_keras_model():
    return tf.keras.models.load_model('coffee_disease_model.h5')

try:
    model = load_keras_model()

    CLASS_NAMES = ['Antracnosis', 'Cercospora / Mancha de Hierro', 'Hoja Sana', 'Roya']
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

def get_groq_recommendations(disease_name):

    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
    if not api_key:
        return "⚠️ Error: No se configuró la clave de API de Groq en st.secrets."
    
    client = Groq(api_key=api_key)
    
    prompt = f"""
    Eres un experto agrónomo especializado en el cultivo de café. La hoja analizada presenta la siguiente enfermedad o condición: '{disease_name}'.
    
    Por favor proporciona una respuesta técnica estructurada estrictamente en 5 puntos numerados:
    1. Diferenciación a simple vista (características visuales clave).
    2. Manejo agronómico preventivo y correctivo (fertilización, poda, fungicidas si aplica).
    3. Consulta o asistencia técnica (cuándo contactar técnicos de IHCAFE o expertos).
    4. Monitoreo y seguimiento (frecuencia de revisión en el cultivo).
    5. Registro y trazabilidad (qué variables climáticas o foliares registrar).

    Mantén un lenguaje profesional, claro, conciso y directo para el agricultor.
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile", # O llama3-8b-8192
            temperature=0.3,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error al consultar la API de Groq: {e}"

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="header-title">Captura de Imagen Foliar</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-sub">Posicione la hoja de café bajo luz natural. El sistema detectará automáticamente signos de Roya, Cercospora o Plagas.</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Subir archivo", "Usar cámara"])
    uploaded_file = None
    
    with tab1:
        file_input = st.file_uploader("Seleccione una imagen", type=["jpg", "jpeg", "png"])
        if file_input:
            uploaded_file = file_input
            
    with tab2:
        camera_input = st.camera_input("Tome una foto")
        if camera_input:
            uploaded_file = camera_input

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_column_width=True, caption="Imagen cargada")

with col_right:
    if uploaded_file is not None:

        img_resized = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
        
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx]) * 100
        detected_disease = CLASS_NAMES[class_idx]

        header_col, score_col = st.columns([2, 1])
        with header_col:
            st.markdown(f"### {detected_disease}")
            st.caption("Censores café/foliar - Detectado recientemente")
        with score_col:
            st.markdown(f'<div class="confidence-score">{confidence:.1f}%</div>', unsafe_allow_html=True)
            st.markdown('<div class="confidence-label">CONFIANZA</div>', unsafe_allow_html=True)
        
        st.divider()
        st.markdown("**ORIENTACIÓN Y MANEJO PREVENTIVO**")
        st.caption("A continuación, recomendaciones técnica y detalladas para el manejo de su afección:")
        
        with st.spinner("Generando orientación técnica con Groq AI..."):
            groq_response = get_groq_recommendations(detected_disease)
            
        st.write(groq_response)
    else:
        st.info("👈 Por favor cargue o tome una foto de la hoja de café en el panel izquierdo para realizar el diagnóstico.")

st.markdown("---")
st.caption("© 2026 AGRODETECT - SOPORTE IHCAFE")
