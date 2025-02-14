import streamlit as st
import requests
import json

# FastAPI endpoint
FASTAPI_URL = "http://localhost:8000/predict"

def call_fastapi_api(inputs):
    headers = {"Content-Type": "application/json"}
    response = requests.post(FASTAPI_URL, headers=headers, json=inputs)

    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"API request failed: {response.status_code}")
        return None

def main():
    st.title("Diamond Price Prediction")
    st.write("Enter diamond characteristics to predict the price.")

    carat = st.number_input("Carat", min_value=0.2, max_value=5.01, value=1.0)
    x = st.number_input("Length (mm)", min_value=0.0, max_value=10.74, value=6.5)
    y = st.number_input("Width (mm)", min_value=0.0, max_value=58.9, value=6.6)
    z = st.number_input("Depth (mm)", min_value=0.0, max_value=31.8, value=4.0)
    cut_encoded = st.slider("Cut Quality (Encoded)", min_value=1, max_value=5, value=3)
    color_encoded = st.slider("Color (Encoded)", min_value=1, max_value=7, value=3)
    clarity_encoded = st.slider("Clarity (Encoded)", min_value=1, max_value=8, value=4)
    depth = st.number_input("Total Depth Percentage", min_value=43.0, max_value=79.0, value=62.5)
    table = st.number_input("Table Width", min_value=43.0, max_value=95.0, value=58.0)

    if st.button("Predict Price"):
        inputs = [{
            "carat": carat,
            "x": x,
            "y": y,
            "z": z,
            "cut_encoded": cut_encoded,
            "color_encoded": color_encoded,
            "clarity_encoded": clarity_encoded,
            "depth": depth,
            "table": table
        }]

        result = call_fastapi_api(inputs)
        if result:
            price = result.get("predicted_price", ["Unknown"])[0]
            st.success(f"Predicted Diamond Price: ${price:,.2f}")

if __name__ == "__main__":
    main()
