import streamlit as st
import pandas as pd
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

# Cargar los datos
df_path = "Energy_consumption.csv"
if os.path.exists(df_path):
    df = pd.read_csv(df_path)
else:
    st.error("El archivo de datos no se encuentra en el directorio.")
    df = pd.DataFrame()

# Mostrar el dataset
st.title("Predicción de Consumo de Energía")
st.write("Este aplicativo permite comparar las predicciones de modelos SVM y Decision Tree.")

# Selección de modo de entrada
if not df.empty:
    option = st.radio("Selecciona el modo de entrada de datos:", ["Seleccionar un registro", "Ingresar manualmente"])

    if option == "Seleccionar un registro":
        index = st.number_input("Selecciona un índice de fila", min_value=0, max_value=len(df)-1, value=0, step=1)
        input_data = df.iloc[index, :-1].values.reshape(1, -1)
    else:
        input_data = []
        for col in ['Timestamp', 'Temperature', 'Humidity', 'SquareFootage', 'Occupancy', 'HVACUsage', 'LightingUsage', 'RenewableEnergy', 'DayOfWeek', 'Holiday']:
            value = st.number_input(f"{col}", value=float(df[col].mean()))
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
