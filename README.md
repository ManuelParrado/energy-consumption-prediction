# **Energy Consumption Prediction Project**

This project implements an energy consumption prediction system using **Scikit-learn**, **Flask**, **Streamlit**, and **Poetry** for dependency management. **Decision Tree (DT)** and **Support Vector Machine (SVM)** models have been developed, trained, evaluated, and serialized for use in two environments:
1. **Flask API** for predictions via an endpoint.
2. **Streamlit interface** for an interactive user experience.

## **Dataset Used**
To train the models, the following **Kaggle** dataset has been selected:

📌 **Energy Consumption Prediction Dataset**  
🔗 Available on Kaggle: [Energy Consumption Prediction](https://www.kaggle.com/datasets/mrsimple07/energy-consumption-prediction)  

This dataset contains information on energy consumption under different environmental and building occupancy conditions, including variables such as:
- **Temperature** (`Temperature`)
- **Humidity** (`Humidity`)
- **Square Footage** (`SquareFootage`)
- **Occupancy** (`Occupancy`)
- **HVAC Usage** (`HVACUsage`)
- **Lighting Usage** (`LightingUsage`)
- **Renewable Energy Used** (`RenewableEnergy`)
- **Day of the Week** (`DayOfWeek`)
- **Whether it is a holiday or not** (`Holiday`)

This dataset allows for the construction of a **regression model** to predict energy consumption based on the given conditions.

## **Project Features**
- **Dependency management with Poetry** for a clean and replicable development environment.
- **Data preparation** including handling missing values, converting categorical attributes, and normalization.
- **Model training with Scikit-learn**, including optimization with **GridSearchCV**.
- **Model evaluation** using metrics such as **R² Score, MAE, and MSE**.
- **Model serialization** for reuse in multiple environments.
- **Flask API creation** to make predictions via HTTP requests.
- **Streamlit graphical interface** to input data and visualize predictions.
