## Explicación del Código Fuente (`app.py`)

El archivo `app.py` gestiona la interfaz gráfica de usuario (UI), el procesamiento de imágenes, la inferencia del modelo de visión artificial y la integración con el LLM de Groq. A continuación se detalla la estructura lógica del programa:

### 1. Configuración General e Inyección de Estilos CSS
- **`st.set_page_config()`**: Define el título de la pestaña web, el ícono representativo y activa el modo de pantalla ancha (`layout="wide"`).
- **CSS Personalizado (`st.markdown`)**: Inyecta reglas de estilo personalizadas para aplicar una paleta de colores agrícola (verde esmeralda, beige y tonos oscuros), tarjetas con sombra (`custom-card`), tipografía *Plus Jakarta Sans* y componentes gráficos destacados como el *Hero Banner* y el *Badge de Confianza*.

### 2. Carga del Modelo de Visión Artificial
- **`load_keras_model()`**: Carga el archivo del modelo entrenado (`coffee_disease_model.h5`) utilizando la librería TensorFlow.
- **`@st.cache_resource`**: Decorador de Streamlit que optimiza la memoria al cargar el modelo una sola vez al iniciar la aplicación, evitando recargarlo en cada interacción del usuario.
- **`CLASS_NAMES`**: Arreglo que asigna las clases del modelo (`Antracnosis`, `Cercospora / Mancha de Hierro`, `Hoja Sana`, `Roya`).

### 3. Integración con la API de Groq
- **`get_groq_recommendations(disease_name)`**:
  - Obtiene de forma segura la clave de API desde los secretos de la plataforma (`st.secrets["GROQ_API_KEY"]`).
  - Utiliza el cliente oficial de Groq con el modelo **`llama-3.3-70b-versatile`** para generar una respuesta agronómica profesional.
  - Implementa un *prompt* técnico parametrizado que exige una estructura estricta en 5 puntos: *Diferenciación visual*, *Manejo agronómico*, *Asistencia técnica*, *Monitoreo* y *Trazabilidad*.

### 4. Captura y Preprocesamiento de Imágenes
- **Pestañas de Selección (`st.tabs`)**: Permite al usuario cargar un archivo (`st.file_uploader`) o capturar una foto directamente desde la cámara web de su dispositivo (`st.camera_input`).
- **Preprocesamiento con PIL y NumPy**:
  - Convierte la imagen cargada al espacio de color RGB.
  - Redimensiona la matriz a la resolución de entrada requerida por la red neuronal ($224 \times 224$ píxeles).
  - Normaliza los valores de los píxeles al rango $[0, 1]$ dividiendo por $255.0$ y agrega la dimensión del lote (`batch_size`).

### 5. Inferencia y Presentación Dinámica de Resultados
- **Inferencia**: Ejecuta `model.predict()` para calcular las probabilidades del diagnóstico y utiliza `np.argmax()` para determinar la enfermedad predicha y su porcentaje de confianza ($0\% - 100\%$).
- **Renderizado Dinámico**:
  - Muestra un *badge* de estado destacado (verde si es 'Hoja Sana' o rojo/alerta si presenta alguna patología).
  - Despliega la respuesta de Groq parseando la salida mediante expresiones regulares (`re.split`) para encapsular cada recomendación dentro de tarjetas visuales independientes (`rec-item`).
