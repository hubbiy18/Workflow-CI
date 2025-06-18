import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import joblib

def main():
    # Load dataset
    df = pd.read_csv("diabetes_cleaned.csv")

    # Pastikan kolom target benar
    if "diabetes" not in df.columns:
        raise ValueError("Kolom target 'diabetes' tidak ditemukan.")

    # Split features and target
    X = df.drop("diabetes", axis=1)
    y = df["diabetes"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_res, y_train_res)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Logging
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)

    # Print report
    print("Classification Report:\n", report)

    # Simpan model
    joblib.dump(model, "model.pkl")
    mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()
