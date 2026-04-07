# Bengaluru-house-price-prediction

A machine learning-powered web application that predicts real estate prices in Bengaluru, India. This project uses a Random Forest Regressor to provide accurate estimates based on location, square footage, number of bedrooms (BHK), and bathrooms.

🚀 Features
Accurate Predictions: Uses an ensemble learning approach (Random Forest) to handle non-linear relationships in real estate data.

Interactive UI: Built with Streamlit, featuring a custom background and a user-friendly sidebar for inputs.

Data Pipeline: Includes a complete preprocessing pipeline with ColumnTransformer and OneHotEncoder to handle categorical location data.

Log Transformation: Implements logarithmic scaling on target variables to handle price skewness and improve model reliability.

🛠️ Tech Stack
Language: Python

ML Library: Scikit-Learn

Data Handling: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Web Framework: Streamlit

📊 Dataset
The dataset contains information about house size, location, and price in Bengaluru.
Key preprocessing steps included:

Handling missing values in location and size.

Converting total_sqft ranges into numeric averages.

Feature engineering to create the bhk column.

Outlier removal (e.g., removing properties where price per sqft was logically inconsistent).

⚙️ Installation & Setup
Clone the repository:

Bash
git clone https://github.com/YourUsername/Bengaluru-House-Price-Prediction.git
cd Bengaluru-House-Price-Prediction
Install dependencies:

Bash
pip install -r requirements.txt
Run the Application:

Bash
streamlit run app.py
🧠 Model Performance
The project compares multiple regression models:

Ridge Regression: Baseline model (~61% accuracy).

Random Forest Regressor: Current production model, offering better handling of location-based variance and non-linear trends.

📂 Project Structure
Plaintext
├── app.py                   # Streamlit web application code
├── banglore_house_pred.ipynb # Jupyter Notebook with EDA and Model Training
├── RFModel.pkl              # Saved Random Forest Pipeline
├── cleaned_house_data.csv   # Cleaned dataset for the UI dropdowns
├── image.jpg                # Background image for the UI
└── requirements.txt         # List of required Python libraries
