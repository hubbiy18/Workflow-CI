import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import os
import joblib

def main():
    df = pd.read_csv("diabetes_cleaned.csv")

    if "diabetes" not in df.columns:
        raise ValueError("Kolom 'diabetes' tidak ditemukan!")

    X = df.drop("diabetes", axis=1)
    y = df["diabetes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("train_score", model.score(X_train, y_train))
        mlflow.log_metric("test_score", model.score(X_test, y_test))

        # Log artifact model
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Simpan model.pkl ke disk (untuk upload ke Google Drive)
        os.makedirs("model", exist_ok=True)
        joblib.dump(model, "model/model.pkl")

        # Log ke MLflow sebagai artifact manual juga
        mlflow.log_artifact("model/model.pkl", artifact_path="model")

if __name__ == "__main__":
    main()
