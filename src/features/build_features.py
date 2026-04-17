import pandas as pd


def _map_binary_series(s: pd.Series) -> pd.Series:
    """
    Apply deterministic binary encoding to 2-category features.
    """
    # Get unique values and remove NaN
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)
    
    # Map Yes/No to 1/0
    if valset == {"Yes", "No"}:
        return s.map({"No": 0, "Yes": 1}).astype("Int64")
        
    # Map Male/Female to 1/0
    if valset == {"Male", "Female"}:
        return s.map({"Female": 0, "Male": 1}).astype("Int64")

    # Else order alphabetical and assign first item 0
    if len(vals) == 2:
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return s.astype(str).map(mapping).astype("Int64")

    return s


def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Feature engineering function:
    - Binary encoding 2-categorical features
    - One-hot encoding multi-categorical features
    """

    df = df.copy()

    print("Building Features...")
    print(f"Current columns: {df.columns}")

    binary_cols = [
        col for col in df.columns
        if df[col].dtype == "object" 
        and df[col].nunique() == 2
    ]

    multi_cols = [
        col for col in df.columns
        if df[col].dtype == "object" 
        and df[col].nunique() > 2
    ]

    # Convert boolean columns to integer
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)

    # Convert binary columns to integer
    for col in binary_cols:
        df[col] = _map_binary_series(df[col].astype(str))
    
    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
    
    # Convert nullable integers to standard integers
    for col in binary_cols:
        if pd.api.types.is_integer_dtype(df[col]):
            # Fill any NaN values with 0 and convert to int
            df[col] = df[col].fillna(0).astype(int)

    return df