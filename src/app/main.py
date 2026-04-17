from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import gradio as gr
import os
import sys


# Allow imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from serving.inference import predict


app = FastAPI(
    title="Telco Customer Churn Prediction API",
)


@app.get("/")
def root():
    """
    Health check endpoint.
    """
    return {"status": "ok"}


class CustomerData(BaseModel):
    """
    Customer data schema.
    """
    gender: str
    SeniorCitizen: int
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
    Prediction endpoint for model.
    """
    try:
        result = predict(data.model_dump())
        return {"prediction": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def format_prediction_html(result) -> str:
    """
    Format model prediction as a colored result card.
    """
    result_text = str(result).strip()
    is_churn = result_text.lower() in {"yes", "churn", "1", "true"}

    bg_color = "#f8d7da" if is_churn else "#d1fae5"
    border_color = "#dc3545" if is_churn else "#198754"
    text_color = "#842029" if is_churn else "#0f5132"
    label = "Likely to Churn" if is_churn else "Not Likely to Churn"

    return f"""
    <div style="
        padding: 14px;
        border-radius: 10px;
        background-color: {bg_color};
        border: 1px solid {border_color};
        color: {text_color};
        font-weight: 600;
        font-size: 16px;
        text-align: center;
        margin-top: 6px;
    ">
        {label}<br>
    </div>
    """


def gradio_interface(
    gender,
    senior_citizen,
    partner,
    dependents,
    contract,
    payment_method,
    paperless_billing,
    multiple_lines,
    internet_service,
    online_security,
    online_backup,
    device_protection,
    phone_service,
    tech_support,
    streaming_tv,
    streaming_movies,
    tenure,
    monthly_charges,
    total_charges,
):
    """
    Gradio interface callback.
    """
    data = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen else 0,
        "Partner": "Yes" if partner else "No",
        "Dependents": "Yes" if dependents else "No",
        "PhoneService": "Yes" if phone_service else "No",
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": "Yes" if paperless_billing else "No",
        "PaymentMethod": payment_method,
        "tenure": int(tenure),
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }

    try:
        result = predict(data)
        return format_prediction_html(result)
    except Exception as exc:
        return f"""
        <div style="
            padding: 14px;
            border-radius: 10px;
            background-color: #f8d7da;
            border: 1px solid #dc3545;
            color: #842029;
            font-weight: 600;
            margin-top: 6px;
        ">
            Error: {exc}
        </div>
        """


with gr.Blocks(theme=gr.themes.Soft(), title="Telco Customer Churn Predictor") as demo:
    gr.Markdown(
        """
        # Telco Customer Churn Predictor
        Fill in the customer details below to get a churn prediction.
        """
    )

    with gr.Row():
        # Column 1: Demographics + Account stacked
        with gr.Column():
            with gr.Group():
                gr.HTML(
                    """
                    <div style="padding: 8px 10px 4px 10px; font-size: 18px; font-weight: 600;">
                        Demographics
                    </div>
                    """
                )

                gender = gr.Radio(
                    ["Male", "Female"],
                    label="Gender",
                    value="Male",
                )

                with gr.Row():
                    senior_citizen = gr.Checkbox(
                        label="Senior Citizen",
                        value=False,
                    )
                    partner = gr.Checkbox(
                        label="Partner",
                        value=False,
                    )
                    dependents = gr.Checkbox(
                        label="Dependents",
                        value=False,
                    )

            with gr.Group():
                gr.HTML(
                    """
                    <div style="padding: 8px 10px 4px 10px; font-size: 18px; font-weight: 600;">
                        Account
                    </div>
                    """
                )

                with gr.Row():
                    contract = gr.Dropdown(
                        ["Month-to-month", "One year", "Two year"],
                        label="Contract",
                        value="Month-to-month",
                    )
                    payment_method = gr.Dropdown(
                        [
                            "Electronic check",
                            "Mailed check",
                            "Bank transfer (automatic)",
                            "Credit card (automatic)",
                        ],
                        label="Payment Method",
                        value="Electronic check",
                    )

                paperless_billing = gr.Checkbox(
                    label="Paperless Billing",
                    value=True,
                )

        # Column 2: Services
        with gr.Column():
            with gr.Group():
                gr.HTML(
                    """
                    <div style="padding: 8px 10px 4px 10px; font-size: 18px; font-weight: 600;">
                        Services
                    </div>
                    """
                )

                with gr.Row():
                    multiple_lines = gr.Dropdown(
                        ["Yes", "No", "No phone service"],
                        label="Multiple Lines",
                        value="No",
                    )
                    internet_service = gr.Dropdown(
                        ["DSL", "Fiber optic", "No"],
                        label="Internet Service",
                        value="Fiber optic",
                    )
                    online_security = gr.Dropdown(
                        ["Yes", "No", "No internet service"],
                        label="Online Security",
                        value="No",
                    )

                with gr.Row():
                    online_backup = gr.Dropdown(
                        ["Yes", "No", "No internet service"],
                        label="Online Backup",
                        value="No",
                    )
                    device_protection = gr.Dropdown(
                        ["Yes", "No", "No internet service"],
                        label="Device Protection",
                        value="No",
                    )

                phone_service = gr.Checkbox(
                    label="Phone Service",
                    value=True,
                )

        # Column 3: More Services
        with gr.Column():
            with gr.Group():
                gr.HTML(
                    """
                    <div style="padding: 8px 10px 4px 10px; font-size: 18px; font-weight: 600;">
                        More Services
                    </div>
                    """
                )

                tech_support = gr.Dropdown(
                    ["Yes", "No", "No internet service"],
                    label="Tech Support",
                    value="No",
                )
                streaming_tv = gr.Dropdown(
                    ["Yes", "No", "No internet service"],
                    label="Streaming TV",
                    value="Yes",
                )
                streaming_movies = gr.Dropdown(
                    ["Yes", "No", "No internet service"],
                    label="Streaming Movies",
                    value="Yes",
                )

        # Column 4: Charges
        with gr.Column():
            with gr.Group():
                gr.HTML(
                    """
                    <div style="padding: 8px 10px 4px 10px; font-size: 18px; font-weight: 600;">
                        Charges
                    </div>
                    """
                )

                tenure = gr.Number(
                    label="Tenure (months)",
                    value=1,
                )
                monthly_charges = gr.Number(
                    label="Monthly Charges ($)",
                    value=85.0,
                )
                total_charges = gr.Number(
                    label="Total Charges ($)",
                    value=85.0,
                )

    with gr.Row():
        predict_button = gr.Button("Predict", variant="primary")
        clear_button = gr.ClearButton()

    output = gr.HTML(label="Churn Prediction")

    predict_button.click(
        fn=gradio_interface,
        inputs=[
            gender,
            senior_citizen,
            partner,
            dependents,
            contract,
            payment_method,
            paperless_billing,
            multiple_lines,
            internet_service,
            online_security,
            online_backup,
            device_protection,
            phone_service,
            tech_support,
            streaming_tv,
            streaming_movies,
            tenure,
            monthly_charges,
            total_charges,
        ],
        outputs=output,
    )

    clear_button.add(
        [
            gender,
            senior_citizen,
            partner,
            dependents,
            contract,
            payment_method,
            paperless_billing,
            multiple_lines,
            internet_service,
            online_security,
            online_backup,
            device_protection,
            phone_service,
            tech_support,
            streaming_tv,
            streaming_movies,
            tenure,
            monthly_charges,
            total_charges,
            output,
        ]
    )


app = gr.mount_gradio_app(app, demo, path="/ui")