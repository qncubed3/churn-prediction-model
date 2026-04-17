import os
import pandas as pd
import mlflow


MODEL_DIR = "mlruns/1/models/m-350da676ed7b42818b2c80b74d5678cb/artifacts"

# Load pretrained model
try:
    # Load the trained XGBoost model in MLflow pyfunc format
    # This ensures compatibility regardless of the underlying ML library
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"✅ Model loaded successfully from {MODEL_DIR}")
except Exception as e:
    print(f"❌ Failed to load model from {MODEL_DIR}: {e}")
    # Fallback for local development (OPTIONAL)
    try:
        # Try loading from local MLflow tracking
        import glob
        local_model_paths = glob.glob("./mlruns/*/*/artifacts/model")
        if local_model_paths:
            latest_model = max(local_model_paths, key=os.path.getmtime)
            model = mlflow.pyfunc.load_model(latest_model)
            MODEL_DIR = latest_model
            print(f"✅ Fallback: Loaded model from {latest_model}")
        else:
            raise Exception("No model found in local mlruns")
    except Exception as fallback_error:
        raise Exception(f"Failed to load model: {e}. Fallback failed: {fallback_error}")

# Declare binary and numerical columns
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},  
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same transformations to features as used in model training
    """

    df = df.copy()

    df.columns = df.columns.str.strip()

    # Ensure correct typing for columns
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(0)

    for col, mapping in BINARY_MAP.items():
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .map(mapping)
                .astype("Int64")
                .fillna(0)
                .astype(int)
            )

    # One-hot encoding
    obj_cols = [col for col in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # Boolean conversion
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)



    return df

def predict(input_dict: dict) -> str:
    """
    Prediction function used by API
    """

    # Transform input dictionary into correct format
    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)

    # Create prediction using pretrained model
    try:
        pred = model.predict(df_enc)
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")
    
    if pred == 1:
        return "Likely to churn"
    else:
        return "Not likely to churn"


