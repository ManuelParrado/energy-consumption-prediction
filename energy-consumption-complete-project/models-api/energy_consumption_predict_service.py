def predict_single(customer, dv, model):
    try:
        x = dv.transform([customer])
        y_pred = model.predict(x)  # Predicción de consumo energético

        # Devolvemos la predicción
        return y_pred[0]
    except Exception as e:
        raise ValueError(f"Error during SVM prediction: {e}")