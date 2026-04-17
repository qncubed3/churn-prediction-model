from fastapi import FastAPI
from pydantic import BaseModel
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from serving.inference import predict

app = FastAPI(
    title="Telco Customer Churn Predictoin API",
)

@app.get("/")
def root():
    """
    Health check endpoint
    """
    return { "status": "ok" }

class CustomerData(BaseModel):
    """
    Customer data schema
    """
    gender: str
    Partner: str
    Dependents: str
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
    tenure: int
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def get_prediction(data: CustomerData):
    """
    Prediction endpoint for model
    """
    try:
        result = predict(data.dict())
        return { "prediction": result }
    except Exception as e:
        return { "error": str(e) }