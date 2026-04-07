import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

# 1. Function to set the local background image
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

            /* Main Container: Reduced opacity and forced Black Text */
            .main .block-container {{
                background-color: rgba(255, 255, 255, 0.5); 
                padding: 3rem;
                border-radius: 15px;
                margin-top: 30px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                color: #000000; /* Forces text to black */
            }}
            
            /* Target all headers, labels, and text inside the main area */
            .main h1, .main h2, .main h3, .main p, .main label, .main .stMarkdown {{
                color: #000000 !important;
            }}

            /* Sidebar styling: Kept dark for contrast with white text */
            [data-testid="stSidebar"] {{
                background-color: rgba(0, 0, 0, 0.7);
            }}
            [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] label {{
                color: white !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning(f"Background image '{image_file}' not found.")

# Apply the background
add_local_bg('image.png')

# 2. Load the Random Forest model and the cleaned data
model = pickle.load(open('RFModel.pkl', 'rb'))
df = pd.read_csv('cleaned_house_data.csv')

# 3. UI Header
st.title("🏡 Bengaluru House Price Predictor")
st.write("Enter property details to estimate the market price in Lakhs.")

# 4. Input Fields in Sidebar
st.sidebar.header("Property Details")
location = st.sidebar.selectbox("Select Location", sorted(df['location'].unique()))
total_sqft = st.sidebar.number_input("Total Square Feet", min_value=300, value=1200, step=50)
bath = st.sidebar.slider("Number of Bathrooms", 1, 10, 2)
bhk = st.sidebar.slider("BHK (Bedrooms)", 1, 10, 2)

# 5. Prediction Logic
if st.button("Predict Price"):
    input_data = pd.DataFrame(
        [[location, total_sqft, bath, bhk]], 
        columns=['location', 'total_sqft', 'bath', 'bhk']
    )
    
    raw_prediction = model.predict(input_data)[0]
    final_price = np.expm1(raw_prediction) 

    # Display Result
    st.subheader(f"Estimated Price: ₹ {np.round(final_price, 2)} Lakhs")
    
    if final_price < 10:
        st.error("Warning: The predicted price seems unusually low.")