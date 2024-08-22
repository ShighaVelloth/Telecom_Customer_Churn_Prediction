from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
from typing import Optional

# Load the trained model
MODEL_PATH = "C:/Users/ragesh/OneDrive/Desktop/PROJECTS/logistic_regression_model.pkl"

def load_model():
    return joblib.load(MODEL_PATH)

# Define the FastAPI app
app = FastAPI()

# Define a Pydantic model for request validation
class CustomerData(BaseModel):
    customerID: Optional[str]
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# Encoding function
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

# Define the prediction endpoint
@app.post("/predict")
def predict_churn(customer_data: CustomerData):
    # Convert Pydantic model to DataFrame
    df = pd.DataFrame([customer_data.dict()])

    # Encode features
    df_encoded, _ = encode_features(df)

    # Load model
    model = load_model()

    # Drop customerID for prediction
    df_encoded = df_encoded.drop(columns=['customerID'])

    # Predict churn probability
    churn_probability = model.predict_proba(df_encoded)[:, 1]

    # Format churn probability
    formatted_churn_probability = "{:.2%}".format(churn_probability.item())
    
    return {"churn_probability": formatted_churn_probability}
