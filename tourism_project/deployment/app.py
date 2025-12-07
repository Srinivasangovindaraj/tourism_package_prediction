import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download
from collections import OrderedDict
import numpy as np

# INITIAL CONFIGURATION
HF_USER_ID = "SriniGS" # HF_USER
MODEL_REPO_ID = "SriniGS/tourism-package-prediction-v2"
MODEL_FILENAME = "xgboost_model_pipeline.joblib"

#  1. Download the model from the Model Hub & 2. Load the model
@st.cache_resource
def load_model_from_hub():
    """Downloads and loads the model pipeline from Hugging Face Model Hub."""
    #st.write(f"Downloading model: {MODEL_FILENAME} from {MODEL_REPO_ID}...")

    # Download the file from the Hugging Face Hub
    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
        repo_type="model"
    )

    # Load the model
    model = joblib.load(model_path)
    return model

# Load the model pipeline once
try:
    model_pipeline = load_model_from_hub()
    # Define the classification threshold (used in training/evaluation)
    CLASSIFICATION_THRESHOLD = 0.45
except Exception as e:
    st.error(f"Error loading model from Hugging Face: {e}")
    st.stop()


# 3. Streamlit UI for tourism Prediction
st.title("Tourism Package Purchase Predictor")
# Batch Prediction
st.subheader("Enter customer details to predict the likelihood of them purchasing the offered tourism package.")

# 4. Collect user input

# 4.1. Define input fields based on the features used in the model
# Using OrderedDict to maintain the input order

input_data = OrderedDict([
    # Numerical features
    ("Age", st.slider("Age", 18, 65, 35)),
    ("CityTier", st.selectbox("City Tier (City importance)", [1, 2, 3], index=2)),
    ("DurationOfPitch", st.slider("Duration of Pitch (minutes)", 1.0, 60.0, 10.0)),
    ("NumberOfPersonVisiting", st.selectbox("Number of People Visiting", [1, 2, 3, 4, 5], index=2)),
    ("NumberOfFollowups", st.slider("Number of Follow-ups", 1.0, 6.0, 3.0)),
    ("PreferredPropertyStar", st.slider("Preferred Property Star Rating", 3.0, 5.0, 3.0)),
    ("NumberOfTrips", st.slider("Number of Previous Trips", 1.0, 20.0, 5.0)),
    ("Passport", st.selectbox("Passport (1=Yes, 0=No)", [0, 1])),
    ("PitchSatisfactionScore", st.slider("Pitch Satisfaction Score (1-5)", 1, 5, 3)),
    ("OwnCar", st.selectbox("Own Car (1=Yes, 0=No)", [0, 1])),
    ("NumberOfChildrenVisiting", st.slider("Number of Children Visiting", 0.0, 5.0, 1.0)),
    ("MonthlyIncome", st.number_input("Monthly Income", 10000.0, 60000.0, 25000.0, step=100.0)),

    # Categorical features
    ("TypeofContact", st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited", "Others"])),
    ("Occupation", st.selectbox("Occupation", ["Salaried", "Small Business", "Government Sector", "Free Lancer"])),
    ("Gender", st.selectbox("Gender", ["Male", "Female", "Fe Male"])),
    ("ProductPitched", st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])),
    ("MaritalStatus", st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])),
    ("Designation", st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "VP", "AVP"])),
])


# --- 5. Prepare input data ---
def prepare_input(input_dict):
    """Converts the OrderedDict input to a single-row DataFrame."""
    # Ensure the columns match the exact order expected by the model's preprocessor
    df_input = pd.DataFrame([input_dict])
    return df_input


# --- 6. Create Predict button on the user interface ---
st.header("Prediction Results")
# Create two columns for structure
col1, col2 = st.columns([1, 2])

if col1.button("Predict Package Purchase", help="Click to see the prediction"):

    with st.spinner('Analyzing customer data...'):
        # 5. Prepare input data
        final_input_df = prepare_input(input_data)

        # Make prediction
        # The model pipeline handles all preprocessing (scaling, one-hot encoding) internally.
        prediction_proba = model_pipeline.predict_proba(final_input_df)[:, 1][0]

        # Use the established classification threshold
        prediction = 1 if prediction_proba >= CLASSIFICATION_THRESHOLD else 0

        # --- Display Results ---

        col2.metric(label="Purchase Probability (Risk Score)",
                    value=f"{prediction_proba * 100:.2f} %")

        if prediction == 1:
            st.success(
                f"Prediction: Likely to Purchase (Threshold > {CLASSIFICATION_THRESHOLD}) "
            )
            st.balloons()
        else:
            st.info(
                f"Prediction: Not Likely to Purchase (Threshold < {CLASSIFICATION_THRESHOLD})"
            )

        st.subheader("Customer Data Summary")
        st.dataframe(final_input_df)
