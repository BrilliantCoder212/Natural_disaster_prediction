import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# Load trained model
model = joblib.load("EMDAT_Disaster_Prediction_Model.pkl")

# Load label encoder for disaster types
label_encoder = LabelEncoder()
label_encoder.classes_ = ['Flood', 'Storm', 'Drought', 'Landslide', 'Earthquake',
                          'Epidemic', 'Wildfire', 'Mass movement (dry)',
                          'Extreme temperature', 'Volcanic activity',
                          'Insect infestation', 'Impact', 'Animal accident', 'Glacial lake outburst']

# Standard scaler for numerical features
scaler = StandardScaler()

def get_location_data():
    """Mock function to fetch necessary features for the user's location."""
    return {
        "Year": 2025,
        "Total Deaths": np.random.randint(0, 100),
        "Total Affected": np.random.randint(0, 100000),
        "Total Damages ('000 US$)": np.random.randint(0, 1000000)
    }

# Streamlit App
st.title("Natural Disaster Prediction Web App")

# Step 1: Get Data Button
if st.button("Get Data"):
    data = get_location_data()
    st.session_state["input_data"] = data
    st.success("Data fetched successfully! Proceed to select a disaster type.")

# Step 2: Disaster Selection
if "input_data" in st.session_state:
    disaster_type = st.selectbox("Select a Disaster Type", label_encoder.classes_)
    st.session_state["selected_disaster"] = disaster_type

# Step 3: Predict Button
if "selected_disaster" in st.session_state:
    if st.button("Predict"):
        # Convert input data to DataFrame
        input_df = pd.DataFrame([st.session_state["input_data"]])
        input_df["Disaster Type Encoded"] = label_encoder.transform([st.session_state["selected_disaster"]])

        # Normalize numerical features
        input_df_scaled = scaler.fit_transform(input_df)

        # Make prediction
        prediction_probabilities = model.predict_proba(input_df_scaled)[0]
        disaster_index = label_encoder.transform([st.session_state["selected_disaster"]])[0]
        likelihood = prediction_probabilities[disaster_index] * 100
        
        st.success(f"Predicted likelihood of {st.session_state['selected_disaster']} in this area: {likelihood:.2f}%")
