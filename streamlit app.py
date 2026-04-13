import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
from PIL import Image

# Import your inference utilities
from src.inference import load_model_and_preprocessing, predict_price

st.set_page_config(page_title=" Multimodal Price Predictor")

st.title(" Property Price Prediction (Multimodal)")
st.markdown("Predict price using **image + structured data**")

# Load model
@st.cache_resource
def load_all():
    model, prep = load_model_and_preprocessing()
    return model, prep

model, prep = load_all()

# Image Upload

st.subheader(" Upload Property Image")
image = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

# Tabular Inputs

st.subheader(" Property Details")

# Dynamically create inputs
tabular_input = {}

for col in prep["numeric_cols"]:
    tabular_input[col] = st.number_input(f"{col}", value=0.0)

for col in prep["categorical_cols"]:
    tabular_input[col] = st.text_input(f"{col}", value="unknown")

# Convert to DataFrame
tabular_df = pd.DataFrame([tabular_input])

# Prediction

if st.button("Predict Price"):

    if image is None:
        st.warning("Please upload an image")
    else:
        pil_image = Image.open(image).convert("RGB")

        try:
            price = predict_price(model, prep, pil_image, tabular_df)

            st.subheader(" Predicted Price")
            st.success(f"${price:,.2f}")

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

# UX Upgrade

st.image(pil_image, caption="Uploaded Property", use_column_width=True)

st.info("Prediction uses both visual and structured signals.")
