from typing import Tuple, List
import pandas as pd

def validate_telco_data(df) -> Tuple[bool, List[str]]:
    print("Validating data...")
    failed = []

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    required_cols = [
        "customerID", "gender", "Partner", "Dependents",
        "PhoneService", "InternetService", "Contract",
        "tenure", "MonthlyCharges", "TotalCharges"
    ]
    for col in required_cols:
        if col not in df.columns:
            failed.append(f"missing column: {col}")

    checks = [
        (df["customerID"].isnull().any(),                                            "customerID has nulls"),
        (df["tenure"].isnull().any(),                                                "tenure has nulls"),
        (df["MonthlyCharges"].isnull().any(),                                        "MonthlyCharges has nulls"),
        (~df["gender"].isin(["Male", "Female"]).all(),                               "gender invalid values"),
        (~df["Partner"].isin(["Yes", "No"]).all(),                                   "Partner invalid values"),
        (~df["Dependents"].isin(["Yes", "No"]).all(),                                "Dependents invalid values"),
        (~df["PhoneService"].isin(["Yes", "No"]).all(),                              "PhoneService invalid values"),
        (~df["Contract"].isin(["Month-to-month", "One year", "Two year"]).all(),     "Contract invalid values"),
        (~df["InternetService"].isin(["DSL", "Fiber optic", "No"]).all(),            "InternetService invalid values"),
        ((df["tenure"] < 0).any(),                                                   "tenure negative"),
        ((df["tenure"] > 120).any(),                                                 "tenure out of range"),
        ((df["MonthlyCharges"] < 0).any(),                                           "MonthlyCharges negative"),
        ((df["MonthlyCharges"] > 200).any(),                                         "MonthlyCharges out of range"),
        ((df["TotalCharges"] < 0).any(),                                             "TotalCharges negative"),
    ]

    for condition, message in checks:
        if condition:
            failed.append(message)

    total = len(checks) + len(required_cols)
    passed = total - len(failed)

    if not failed:
        print(f"Validation passed: {passed}/{total} checks")
    else:
        print(f"Validation failed: {len(failed)}/{total} checks failed — {failed}")

    return len(failed) == 0, failed