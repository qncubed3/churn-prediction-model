import pandas as pd

def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Data cleaning and preprocessing for Telco churn dataset:
    - trim column names
    - drops id column
    - change TotalCharges to numeric
    """

    # Trim headers
    df.columns = df.columns.str.strip()

    # Drop id column
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Map target to 0/1 if it's Yes/No
    if target_col in df.columns and df[target_col].dtype == "object":
        df[target_col] = df[target_col].str.strip().map({"No": 0, "Yes": 1})

    # Change TotalCharges to numeric type
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # SeniorCitizen should be 0/1 ints if present
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)

    # Fill empty numeric cells with 0
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df

    