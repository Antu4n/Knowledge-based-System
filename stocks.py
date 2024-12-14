import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("stocks.pkl")

# Streamlit App
st.title("Stock Price Prediction")

# Input fields for the selected features
day_high = st.number_input("Day High Price", value=0.0)
day_low = st.number_input("Day Low Price", value=0.0)
previous_price = st.number_input("Previous Price", value=0.0)

# Prediction button
if st.button("Predict"):
    # Prepare input for prediction
    input_data = np.array([[day_high, day_low, previous_price]])
    prediction = model.predict(input_data)
    st.write(f"Predicted End of Day Price: {prediction[0]:.2f}")
