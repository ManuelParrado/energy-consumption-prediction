import streamlit as st
import pickle
import os

# Cargar los modelos
@st.cache_resource
def load_models():
    try:
        dt_model_path = "models/dt-energy-consumption-model.pck"
        svm_model_path = "models/svm-energy-consumption-model.pck"
        
        if not os.path.exists(dt_model_path) or not os.path.exists(svm_model_path):
            st.error("Los archivos de modelo no se encuentran en la carpeta 'models'.")
            return None, None
        
        with open(dt_model_path, "rb") as f:
            dt_model = pickle.load(f)
        with open(svm_model_path, "rb") as f:
            svm_model = pickle.load(f)
        return dt_model, svm_model
    except Exception as e:
        st.error(f"Error cargando los modelos: {e}")
        return None, None

dt_model, svm_model = load_models()

# Título de la aplicación
st.title("Predicción de Consumo de Energía")
st.write("Este aplicativo permite comparar las predicciones de modelos SVM y Decision Tree.")

# Formulario de entrada manual
st.header("Introduce los datos para la predicción:")

temperature = st.number_input("Temperatura (°C)", min_value=-50.0, max_value=50.0, step=0.1)
humidity = st.number_input("Humedad (%)", min_value=0, max_value=100, step=1)
square_footage = st.number_input("Metros cuadrados", min_value=10, max_value=10000, step=10)
occupancy = st.number_input("Número de ocupantes", min_value=0, max_value=100, step=1)
hvac_usage = st.selectbox("Uso de HVAC", ["Off", "On"])
lighting_usage = st.selectbox("Uso de iluminación", ["Off", "On"])
renewable_energy = st.number_input("Energía renovable (kWh)", min_value=0, max_value=10000, step=10)
day_of_week = st.selectbox("Día de la semana", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
holiday = st.selectbox("¿Es día festivo?", ["No", "Yes"])

# Convertir valores categóricos a numéricos
hvac_usage_val = 1 if hvac_usage == "On" else 0
lighting_usage_val = 1 if lighting_usage == "On" else 0
holiday_val = 1 if holiday == "Yes" else 0
day_of_week_val = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week)

# Crear lista de entrada para el modelo
input_data = [[temperature, humidity, square_footage, occupancy, hvac_usage_val, lighting_usage_val, renewable_energy, day_of_week_val, holiday_val]]

# Botón de predicción
if st.button("Predecir"):
    if dt_model is not None and svm_model is not None:
        dt_prediction = dt_model.predict(input_data)[0]
        svm_prediction = svm_model.predict(input_data)[0]

        # Mostrar resultados
        st.subheader("Resultados de Predicción")
        st.write(f"**Decision Tree Prediction:** {dt_prediction}")
        st.write(f"**SVM Prediction:** {svm_prediction}")
    else:
        st.warning("Los modelos no fueron cargados correctamente.")
