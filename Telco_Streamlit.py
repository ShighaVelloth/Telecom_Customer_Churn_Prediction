import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# Path of the trained model
MODEL_PATH = "C:/Users/shigha/OneDrive/Desktop/PROJECTS/logistic_regression_model.pkl"

# Load the trained model
def load_model():
    model = joblib.load(MODEL_PATH)
    return model

# Encode categorical features
def encode_features(df):
    label_encoders = {}
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                           'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                           'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                           'PaperlessBilling', 'PaymentMethod']
    
    for column in categorical_columns:
        if column in df.columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
            label_encoders[column] = le
    
    return df, label_encoders

# Predict churn probability
def predict_churn(model, customer_data):
    # Drop customerID for prediction
    customer_data = customer_data.drop(columns=['customerID'])
    churn_probability = model.predict_proba(customer_data)[:, 1]
    return churn_probability

# Streamlit app
st.title("Telco Customer Churn Prediction")

# Load model
model = load_model()

# Input fields
customerID = st.text_input("Customer ID")
gender = st.selectbox("Gender:", ["Female", "Male"])
senior_citizen = st.number_input("SeniorCitizen (0: No, 1: Yes)", min_value=0, max_value=1, step=1)
partner = st.selectbox("Partner:", ["No", "Yes"])
dependents = st.selectbox("Dependents:", ["No", "Yes"])
tenure = st.number_input("Tenure:", min_value=0, step=1)
phone_service = st.selectbox("PhoneService:", ["No", "Yes"])
multiple_lines = st.selectbox("MultipleLines:", ["No", "Yes"])
internet_service = st.selectbox("InternetService:", ["No", "DSL", "Fiber optic"])
online_security = st.selectbox("OnlineSecurity:", ["No", "Yes"])
online_backup = st.selectbox("OnlineBackup:", ["No", "Yes"])
device_protection = st.selectbox("DeviceProtection:", ["No", "Yes"])
tech_support = st.selectbox("TechSupport:", ["No", "Yes"])
streaming_tv = st.selectbox("StreamingTV:", ["No", "Yes"])
streaming_movies = st.selectbox("StreamingMovies:", ["No", "Yes"])
contract = st.selectbox("Contract:", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("PaperlessBilling", ["No", "Yes"])
payment_method = st.selectbox("PaymentMethod:", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, step=0.01)
total_charges = st.number_input("Total Charges", min_value=0.0, step=0.01)

# Confirm button
confirmation_button = st.button("Predict Churn Probability")

if confirmation_button:
    # Create DataFrame for new customer
    new_customer_data = pd.DataFrame({
        "customerID": [customerID],
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    # Encode features
    new_customer_data_encoded, _ = encode_features(new_customer_data)

    # Predict churn probability
    churn_probability = predict_churn(model, new_customer_data_encoded)
    
    # Display results
    formatted_churn_probability = "{:.2%}".format(churn_probability.item())
    st.markdown(f"<h1>Churn Probability: {formatted_churn_probability}</h1>", unsafe_allow_html=True)
    st.write(new_customer_data_encoded.to_dict())
