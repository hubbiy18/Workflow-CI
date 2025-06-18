import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

def main():
    df = pd.read_csv("diabetes_cleaned.csv")
    
    if "diabetes" not in df.columns:
        raise ValueError("Kolom 'diabetes' tidak ditemukan di dataset")

    X = df.drop("diabetes", axis=1)
    y = df["diabetes"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    with mlflow.start_run() as run:
        # Log model ke MLflow
        mlflow.sklearn.log_model(model, "model")

        # Simpan model secara lokal juga, untuk docker build
        mlflow.sklearn.save_model(model, "MLProject/model")

        print("Run ID:", run.info.run_id)

if __name__ == "__main__":
    main()
