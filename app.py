import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model (pipeline)
with open("predictor_model.pkl", "rb") as model_file:
    pipe = pickle.load(model_file)

# Load the OneHotEncoder for categorical variables (if needed)
with open("final_encoders.pkl", "rb") as encoder_file:
    ohe = pickle.load(encoder_file)

st.title("🚗 Car Price Prediction")
st.markdown("### Get an accurate price estimate for your car by entering the details below.")

# User Input Fields
car_model = st.text_input("Car Model:", "Maruti Suzuki Swift")
company = st.text_input("Company:", "Maruti")
year = st.number_input("Year:", min_value=1990, max_value=2025, value=2019)
km_driven = st.number_input("Kilometers Driven:", min_value=0, max_value=500000, value=100)
fuel_type = st.selectbox("Fuel Type:", ["Petrol", "Diesel", "CNG"])

# Create a DataFrame to match the model's expected input format
input_data = pd.DataFrame(
    data=np.array([[
        car_model, company, year, km_driven, fuel_type
    ]]),
    columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
)

# Use the trained pipeline to make the prediction
if st.button("🔮 Predict Price"):
    try:
        prediction = pipe.predict(input_data)
        st.success(f"💰 Estimated Price: **${prediction[0]:,.2f}**")
    except Exception as e:
        st.error(f"⚠️ Prediction Error: {e}")
