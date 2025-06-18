import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    # Set experiment
    mlflow.set_experiment("Diabetes Prediction")

    # Start MLflow run
    with mlflow.start_run():
        # Load dataset
        df = pd.read_csv("diabetes_cleaned.csv")

        # Pisahkan fitur dan label
        X = df.drop("diabetes", axis=1)
        y = df["diabetes"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Inisialisasi dan latih model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Prediksi dan evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Log parameter & metric
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)

        # Log model ke MLflow
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="diabetes_rf_model")

        # Simpan model.pkl ke local
        joblib.dump(model, "model.pkl")

        # Log file model.pkl juga sebagai artifact (untuk upload ke GDrive)
        mlflow.log_artifact("model.pkl", artifact_path="model")

        # Optional: log classification report sebagai txt
        report = classification_report(y_test, y_pred)
        with open("report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("report.txt")

        print("Model trained and logged to MLflow.")

if __name__ == "__main__":
    main()
