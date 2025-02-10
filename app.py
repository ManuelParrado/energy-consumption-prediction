import streamlit as st
import pandas as pd
import pickle

# Cargar los modelos
@st.cache_resource
def load_models():
    with open("dt-energy-consumption-model.pck", "rb") as f:
        dt_model = pickle.load(f)
    with open("svm-energy-consumption-model.pck", "rb") as f:
        svm_model = pickle.load(f)
    return dt_model, svm_model

dt_model, svm_model = load_models()

# Cargar los datos
df = pd.read_csv("Energy_consumption.csv")

# Mostrar el dataset
st.title("Predicción de Consumo de Energía")
st.write("Este aplicativo permite comparar las predicciones de modelos SVM y Decision Tree.")

# Selección de modo de entrada
option = st.radio("Selecciona el modo de entrada de datos:", ["Seleccionar un registro", "Ingresar manualmente"])

if option == "Seleccionar un registro":
    index = st.number_input("Selecciona un índice de fila", min_value=0, max_value=len(df)-1, value=0, step=1)
    input_data = df.iloc[index, :-1].values.reshape(1, -1)
else:
    input_data = []
    for col in df.columns[:-1]:
        value = st.number_input(f"{col}", value=float(df[col].mean()))
        input_data.append(value)
    input_data = [input_data]

# Predicción con los modelos
if st.button("Predecir"):
    dt_prediction = dt_model.predict(input_data)[0]
    svm_prediction = svm_model.predict(input_data)[0]
    
    st.write("### Resultados de Predicción")
    st.write(f"**Decision Tree Prediction:** {dt_prediction}")
    st.write(f"**SVM Prediction:** {svm_prediction}")