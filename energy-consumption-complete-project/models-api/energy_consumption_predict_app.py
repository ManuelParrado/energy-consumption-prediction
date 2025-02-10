from flask import Flask, jsonify, request
import pickle

from energy_consumption_predict_service import predict_single

app = Flask('energy-consumption-predict')

# Cargamos los modelos y el vectorizador
with open('models/svm-energy-consumption-model.pck', 'rb') as f:
    dv, svm_model = pickle.load(f)

with open('models/dt-energy-consumption-model.pck', 'rb') as f:
    _, dt_model = pickle.load(f)
    
print("Modelo Decision Tree cargado:", dt_model)
print("Parámetros del modelo:", dt_model.get_params())

# Función para predecir el consumo energético, mediante el modelo SVM
@app.route('/svm_predict', methods=['POST'])
def svm_predict():
    customer = request.get_json()
    try:
        # Obtener la predicción del modelo SVM
        prediction = predict_single(customer, dv, svm_model)

        result = {
            'model': 'Support Vector Machine',
            'energy_consumption': prediction
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Función para predecir el consumo energético, mediante el modelo DT
@app.route('/dt_predict', methods=['POST'])
def tree_predict():
    customer = request.get_json()
    try:
        # Obtener la predicción del modelo Decision Tree
        prediction = predict_single(customer, dv, dt_model)

        result = {
            'model': 'Decision Tree',
            'energy_consumption': prediction
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)
