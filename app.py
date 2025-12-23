import streamlit as st
import pickle
import json
import numpy as np

# Load model
with open("bangalore_home_prices_model.pickle", "rb") as f:
    model = pickle.load(f)

# Load columns
with open("columns.json", "r") as f:
    data = json.load(f)
    columns = data["data_columns"]

st.title("🏠 Bangalore House Price Prediction")

st.write("Enter house details to predict the price")

# Inputs
total_sqft = st.number_input("Total Square Feet", min_value=300.0, max_value=10000.0, step=50.0)
bath = st.number_input("Bathrooms", min_value=1, max_value=10, step=1)
bhk = st.number_input("BHK", min_value=1, max_value=10, step=1)

location = st.selectbox(
    "Location",
    columns[3:]  # skipping total_sqft, bath, bhk
)

if st.button("Predict Price"):
    x = np.zeros(len(columns))

    x[0] = total_sqft
    x[1] = bath
    x[2] = bhk

    if location in columns:
        loc_index = columns.index(location)
        x[loc_index] = 1

    price = model.predict([x])[0]

    st.success(f"💰 Estimated Price: ₹ {round(price, 2)} Lakhs")
