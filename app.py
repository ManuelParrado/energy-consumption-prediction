import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

with open("models/svm-energy-consumption-model.pck", "rb") as f:
    dv, svm-model = pickle.load(f)

with open("models/dt-energy-consumption-model.pck", "rb") as f:
    dv, dt-model = pickle.load(f)

# Título de la aplicación
st.title("Predicción de Consumo de Energía")

# Formulario para introducir datos
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

# Botón de predicción
if st.button("Predecir"):
    # Crear un diccionario con los datos
    input_data = {
        "Temperature": temperature,
        "Humidity": humidity,
        "SquareFootage": square_footage,
        "Occupancy": occupancy,
        "HVACUsage": hvac_usage,
        "LightingUsage": lighting_usage,
        "RenewableEnergy": renewable_energy,
        "DayOfWeek": day_of_week,
        "Holiday": holiday
    }

    # Transformar los datos del usuario usando DictVectorizer
    X_input = dv.transform([input_data])
    
    svm_prediction = svm-model.predict(X_input)
    dt_prediction = dt-model.predict(X_input)

    # Mostrar resultado
    st.subheader("Resultados de Predicción:")
    st.write(f"**SVM Prediction:** {svm_prediction}")
    st.write(f"**DT Prediction:** {dt_prediction}")
