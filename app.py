import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import zipfile
import os

# -----------------------------
# Base Directory
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# Extract Model ZIP
# -----------------------------
zip_path = os.path.join(BASE_DIR, "RFModel.zip")
model_path = os.path.join(BASE_DIR, "RFModel.pkl")

if not os.path.exists(model_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(BASE_DIR)

# -----------------------------
# Function to Set Background
# -----------------------------
def add_local_bg(image_file):
    try:
        with open(image_file, "rb") as image:
            encoded_string = base64.b64encode(image.read()).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded_string}");
                background-attachment: fixed;
                background-size: cover;
            }}

            /* Main Container */
            .main .block-container {{
                background-color: rgba(255, 255, 255, 0.5);
                padding: 3rem;
                border-radius: 15px;
                margin-top: 30px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                color: #000000;
            }}

            /* Text Styling */
            .main h1,
            .main h2,
            .main h3,
            .main p,
            .main label,
            .main .stMarkdown {{
                color: #000000 !important;
            }}

            /* Sidebar */
            [data-testid="stSidebar"] {{
                background-color: rgba(0, 0, 0, 0.7);
            }}

            [data-testid="stSidebar"] .stMarkdown p,
            [data-testid="stSidebar"] label {{
                color: white !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    except FileNotFoundError:
        st.warning(f"Background image '{image_file}' not found.")


# -----------------------------
# Apply Background
# -----------------------------
image_path = os.path.join(BASE_DIR, "image.png")
add_local_bg(image_path)

# -----------------------------
# Load Model and Dataset
# -----------------------------
model = pickle.load(open(model_path, "rb"))

csv_path = os.path.join(BASE_DIR, "cleaned_house_data.csv")
df = pd.read_csv(csv_path)

# -----------------------------
# App Title
# -----------------------------
st.title("🏡 Bengaluru House Price Predictor")

st.write(
    "Enter property details to estimate the market price in Lakhs."
)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Property Details")

location = st.sidebar.selectbox(
    "Select Location",
    sorted(df['location'].unique())
)

total_sqft = st.sidebar.number_input(
    "Total Square Feet",
    min_value=300,
    value=1200,
    step=50
)

bath = st.sidebar.slider(
    "Number of Bathrooms",
    1,
    10,
    2
)

bhk = st.sidebar.slider(
    "BHK (Bedrooms)",
    1,
    10,
    2
)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Price"):

    input_data = pd.DataFrame(
        [[location, total_sqft, bath, bhk]],
        columns=['location', 'total_sqft', 'bath', 'bhk']
    )

    try:
        raw_prediction = model.predict(input_data)[0]

        # Reverse log transformation
        final_price = np.expm1(raw_prediction)

        st.subheader(
            f"Estimated Price: ₹ {np.round(final_price, 2)} Lakhs"
        )

        if final_price < 10:
            st.error(
                "Warning: The predicted price seems unusually low."
            )

        elif final_price > 1000:
            st.warning(
                "Luxury Property Detected 🚀"
            )

        else:
            st.success(
                "Prediction generated successfully ✅"
            )

    except Exception as e:
        st.error(f"Prediction Error: {e}")
