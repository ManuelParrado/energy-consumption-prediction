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

# Mostrar la aplicación
st.title("Predicción de Consumo de Energía")
st.write("Este aplicativo permite comparar las predicciones de modelos SVM y Decision Tree.")

# Formulario de entrada manual
st.write("### Ingresa los datos de entrada para la predicción")
input_data = []

fields = ["Temperature", "Humidity", "SquareFootage", "Occupancy", "HVACUsage", "LightingUsage", "RenewableEnergy", "DayOfWeek", "Holiday"]
for col in fields:
    value = st.number_input(f"{col}", value=0.0)
    input_data.append(value)
input_data = [input_data]

# Predicción con los modelos
if dt_model is not None and svm_model is not None:
    if st.button("Predecir"):
        dt_prediction = dt_model.predict(input_data)[0]
        svm_prediction = svm_model.predict(input_data)[0]
        
        st.write("### Resultados de Predicción")
        st.write(f"**Decision Tree Prediction:** {dt_prediction}")
        st.write(f"**SVM Prediction:** {svm_prediction}")
else:
    st.warning("Los modelos no fueron cargados correctamente.")
