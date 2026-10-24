import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("house_model.pkl", "rb") as file:
    model = pickle.load(file)

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠"
)

# App heading
st.title("🏠 House Price Prediction")
st.write("Predict the estimated price of a house using Machine Learning.")

# User inputs
area = st.number_input(
    "House Area (Square Feet)",
    min_value=100,
    max_value=10000,
    value=1200
)

bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=2
)

bathrooms = st.number_input(
    "Number of Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)

# Prediction button
if st.button("Predict House Price"):

    input_data = pd.DataFrame(
        [[area, bedrooms, bathrooms]],
        columns=["Area", "Bedrooms", "Bathrooms"]
    )

    predicted_price = model.predict(input_data)[0]

    if predicted_price > 0:
        st.success(
            f"Estimated House Price: ₹{predicted_price:,.0f}"
        )
    else:
        st.warning(
            "The model returned an unrealistic estimate. "
            "Please try different inputs."
        )

st.caption(
    "Educational project. Predictions are based on sample data."
)