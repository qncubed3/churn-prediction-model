#!/usr/bin/env python3
import os, sys, time, argparse, json, joblib
import mlflow, mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_telco_data

def main(args):

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():
        mlflow.log_params({"model": "xgboost", "threshold": args.threshold, "test_size": args.test_size})

        # Load
        df = load_data(args.input)
        print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} cols")

        # Validate
        is_valid, failed = validate_telco_data(df)
        mlflow.log_metric("data_quality_pass", int(is_valid))
        if not is_valid:
            mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expectations.json")
            raise ValueError(f"Validation failed: {failed}")
        print("Validation passed")

        # Preprocess
        df = preprocess_data(df)
        processed_path = os.path.join(project_root, "data", "processed", "telco_churn_processed.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)

        # Feature engineering
        if args.target not in df.columns:
            raise ValueError(f"Target column '{args.target}' not found")
        df_enc = build_features(df, target_col=args.target)
        for c in df_enc.select_dtypes(include=["bool"]).columns:
            df_enc[c] = df_enc[c].astype(int)
        print(f"Features built: {df_enc.shape[1]} columns")

        # Save feature metadata
        feature_cols = list(df_enc.drop(columns=[args.target]).columns)
        artifacts_dir = os.path.join(project_root, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        json.dump(feature_cols, open(os.path.join(artifacts_dir, "feature_columns.json"), "w"))
        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")
        joblib.dump({"feature_columns": feature_cols, "target": args.target},
                    os.path.join(artifacts_dir, "preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))

        # Split
        X, y = df_enc.drop(columns=[args.target]), df_enc[args.target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=42)
        print(f"Split: {X_train.shape[0]} train / {X_test.shape[0]} test")

        # Train
        model = XGBClassifier(
            n_estimators=540, 
            learning_rate=0.05456810322015402, 
            max_depth=4,
            subsample=0.9935442672619198, 
            colsample_bytree=0.8539355014882412,
            min_child_weight=10,
            gamma=4.56117332896945,
            reg_alpha=4.58206937567894,
            reg_lambda=1.1628406059579985,
            n_jobs=-1, 
            random_state=42, 
            eval_metric="logloss",
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
        )
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("train_time", train_time)
        print(f"Trained in {train_time:.2f}s")

        # Evaluate
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= args.threshold).astype(int)
        metrics = {
            "precision": precision_score(y_test, y_pred),
            "recall":    recall_score(y_test, y_pred),
            "f1":        f1_score(y_test, y_pred),
            "roc_auc":   roc_auc_score(y_test, proba),
        }
        mlflow.log_metrics(metrics)
        print(f"Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | "
              f"F1: {metrics['f1']:.3f} | AUC: {metrics['roc_auc']:.3f}")
        print(classification_report(y_test, y_pred, digits=3))

        # Save model
        mlflow.sklearn.log_model(model, artifact_path="model")
        print("Model saved to MLflow")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",      required=True)
    p.add_argument("--target",     default="Churn")
    p.add_argument("--threshold",  type=float, default=0.35)
    p.add_argument("--test_size",  type=float, default=0.2)
    p.add_argument("--experiment", default="Telco Churn")
    p.add_argument("--mlflow_uri", default=None)
    main(p.parse_args())