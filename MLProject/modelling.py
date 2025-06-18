import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def main():
    # Load dataset
    df = pd.read_csv("diabetes_cleaned.csv")

    # Pisahkan fitur dan label
    X = df.drop("diabetes", axis=1)
    y = df["diabetes"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Atur eksperimen MLflow secara dinamis
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "Diabetes Prediction")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Model training
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict dan evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Log parameter dan metrik
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)

        # Simpan model secara lokal
        os.makedirs("outputs", exist_ok=True)
        model_path = "outputs/model.pkl"
        joblib.dump(model, model_path)

        # Log model ke MLflow
        mlflow.sklearn.log_model(model, artifact_path="model")

        print("Model trained and logged to MLflow.")

if __name__ == "__main__":
    main()
