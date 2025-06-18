import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def main():
    # Mulai experiment MLflow
    mlflow.start_run()

    # Load dataset
    df = pd.read_csv("diabetes_cleaned.csv")

    # Pisahkan fitur dan target
    X = df.drop("diabetes", axis=1)
    y = df["diabetes"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inisialisasi model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train model
    clf.fit(X_train, y_train)

    # Prediksi
    y_pred = clf.predict(X_test)

    # Hitung akurasi
    accuracy = accuracy_score(y_test, y_pred)

    # Logging ke MLflow
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", accuracy)

    # Simpan model
    joblib.dump(clf, "model.pkl")
    mlflow.log_artifact("model.pkl", artifact_path="model")

    # Akhiri run
    mlflow.end_run()

if __name__ == "__main__":
    main()
