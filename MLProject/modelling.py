# modelling.py
import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    # Logging setup
    mlflow.set_tracking_uri("file:///" + os.path.abspath("mlruns"))
    mlflow.set_experiment("Diabetes Prediction")

    with mlflow.start_run():
        # Load data
        df = pd.read_csv("../diabetes_cleaned.csv") if os.path.exists("../diabetes_cleaned.csv") else pd.read_csv("diabetes_cleaned.csv")
        X = df.drop("diabetes", axis=1)
        y = df["diabetes"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Log metrics and model
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        # Save manually
        os.makedirs("artifact/model", exist_ok=True)
        joblib.dump(model, "artifact/model/model.pkl")
        print("Model saved to artifact/model/model.pkl")

if __name__ == "__main__":
    main()
