import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import re
from groq import Groq

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AGRODETECT - Detección Foliar Café",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CSS PERSONALIZADO (DISEÑO MODERNO Y LLAMATIVO)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Importación de fuentes */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Fondo de la app */
    .stApp {
        background: linear-gradient(135deg, #f4f7f4 0%, #eef2ed 100%);
    }

    /* Banner Superior Hero */
    .hero-container {
        background: linear-gradient(90deg, #1b3b22 0%, #2d5a37 100%);
        padding: 24px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px rgba(27, 59, 34, 0.15);
    }
    .hero-title {
        font-size: 28px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .hero-subtitle {
        font-size: 14px;
        color: #a3d9b1;
        margin-top: 6px;
        font-weight: 400;
    }

    /* Contenedor tipo Tarjeta */
    .custom-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(0, 0, 0, 0.05);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
    }

    /* Títulos de sección */
    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #1b3b22;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Badge de Diagnóstico */
    .disease-badge {
        background-color: #fde8e8;
        color: #9b1c1c;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
        border: 1px solid #fbd5d5;
    }

    /* Bloque de Confianza */
    .confidence-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        padding: 12px;
        border-radius: 12px;
        text-align: center;
    }
    .confidence-value {
        font-size: 32px;
        font-weight: 800;
        color: #15803d;
        line-height: 1;
    }
    .confidence-text {
        font-size: 11px;
        font-weight: 700;
        color: #166534;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-top: 4px;
    }

    /* Tarjetas de recomendaciones de Groq */
    .rec-item {
        background-color: #f9fafb;
        border-left: 4px solid #2d5a37;
        padding: 14px 16px;
        border-radius: 0 10px 10px 0;
        margin-bottom: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
    }
    .rec-number {
        background-color: #1b3b22;
        color: #ffffff;
        font-weight: 700;
        font-size: 12px;
        padding: 3px 8px;
        border-radius: 6px;
        margin-right: 8px;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #6b7280;
        font-size: 13px;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HERO BANNER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🍃 AGRODETECT <span style="font-weight:300; font-size: 20px;">| Detección Foliar de Café</span></div>
    <div class="hero-subtitle">Sistema inteligente con Diagnóstico por Visión Artificial y Asistente Agrónomo IA (Groq)</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CARGA DEL MODELO
# -----------------------------------------------------------------------------
@st.cache_resource
def load_keras_model():
    return tf.keras.models.load_model('coffee_disease_model.h5')

try:
    model = load_keras_model()
    CLASS_NAMES = ['Antracnosis', 'Cercospora / Mancha de Hierro', 'Hoja Sana', 'Roya']
except Exception as e:
    st.error(f"⚠️ Error al cargar el modelo de IA: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# FUNCIÓN DE CONSULTA A GROQ API
# -----------------------------------------------------------------------------
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
            model="llama-3.3-70b-versatile",
            temperature=0.3,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error al consultar la API de Groq: {e}"

# -----------------------------------------------------------------------------
# ESTRUCTURA DE LA INTERFAZ
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1.2], gap="large")

# --- COLUMNA IZQUIERDA: ENTRADA DE IMAGEN ---
with col_left:
    st.markdown('<div class="section-header">📷 Captura de Imagen Foliar</div>', unsafe_allow_html=True)
    st.caption("Suba una fotografía clara de la hoja de café bajo buena luz natural.")
    
    tab1, tab2 = st.tabs(["📁 Subir Archivo", "📸 Usar Cámara"])
    uploaded_file = None
    
    with tab1:
        file_input = st.file_uploader("Seleccione una imagen (JPG, PNG)", type=["jpg", "jpeg", "png"])
        if file_input:
            uploaded_file = file_input
            
    with tab2:
        camera_input = st.camera_input("Capturar con la cámara")
        if camera_input:
            uploaded_file = camera_input

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_column_width=True, caption="Imagen cargada correctamente")

# --- COLUMNA DERECHA: RESULTADOS Y RECOMENDACIONES ---
with col_right:
    st.markdown('<div class="section-header">📊 Diagnóstico y Recomendaciones</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Preprocesamiento y Predicción
        img_resized = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
        
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx]) * 100
        detected_disease = CLASS_NAMES[class_idx]

        # Tarjeta de Diagnóstico Principal
        res_col1, res_col2 = st.columns([2, 1])
        
        with res_col1:
            st.markdown("##### Condición Detectada:")
            if detected_disease == 'Hoja Sana':
                st.markdown(f'<div class="disease-badge" style="background-color: #dcfce7; color: #166534; border-color: #bbf7d0;">🌿 {detected_disease}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="disease-badge">⚠️ {detected_disease}</div>', unsafe_allow_html=True)
            st.caption("Análisis procesado mediante Red Neuronal Convencional")
            
        with res_col2:
            st.markdown(f"""
            <div class="confidence-box">
                <div class="confidence-value">{confidence:.1f}%</div>
                <div class="confidence-text">Confianza</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr style='margin: 20px 0; border-color: #e5e7eb;'>", unsafe_allow_html=True)
        st.markdown("##### 💡 Orientación Técnica y Recomendaciones (Groq AI)")
        
        with st.spinner("Generando plan de manejo agronómico personalizado..."):
            groq_response = get_groq_recommendations(detected_disease)
        
        # Formateo visual llamativo para la respuesta de Groq
        points = re.split(r'\n(?=\d+\.)', groq_response.strip())
        if len(points) > 1:
            for pt in points:
                if pt.strip():
                    st.markdown(f'<div class="rec-item">{pt.strip()}</div>', unsafe_allow_html=True)
        else:
            st.info(groq_response)

    else:
        st.markdown("""
        <div style="text-align: center; padding: 40px 20px; background: white; border-radius: 12px; border: 2px dashed #d1d5db;">
            <p style="font-size: 40px; margin-bottom: 10px;">👈</p>
            <p style="font-weight: 600; color: #374151;">Esperando imagen...</p>
            <p style="font-size: 13px; color: #6b7280;">Cargue o tome una foto de la hoja de café en el panel izquierdo para obtener el diagnóstico.</p>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="footer">
    © 2026 <b>AGRODETECT</b> — Plataforma Agrónoma con IA | Soporte para cultivo de café
</div>
""", unsafe_allow_html=True)
