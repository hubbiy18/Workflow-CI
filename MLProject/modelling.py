import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import joblib
import os
import argparse

def run_modelling(cleaned_filepath="diabetes_cleaned.csv", model_output="rf_model.pkl", n_estimators=100):
    mlflow.sklearn.autolog(disable=True)

    df = pd.read_csv(cleaned_filepath)
    df = df.astype({col: 'float64' for col in df.select_dtypes(include='int').columns})

    X = df.drop("diabetes", axis=1)
    y = df["diabetes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("Distribusi label setelah SMOTE (training):")
    print(pd.Series(y_train_res).value_counts())

    model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=n_estimators)

    with mlflow.start_run():
        model.fit(X_train_res, y_train_res)

        y_pred = model.predict(X_test)
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=4))

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("smote", True)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", model.score(X_test, y_test))

        # Save as .pkl for external use (like Google Drive)
        joblib.dump(model, model_output)
        print(f"\nModel disimpan ke: {model_output}")

        # Log model.pkl ke MLflow artifacts
        mlflow.log_artifact(model_output)

        # Log MLflow model format (for Docker & serving)
        input_example = X_test.iloc[:5]
        mlflow.sklearn.log_model(model, "model", input_example=input_example)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()

    run_modelling(n_estimators=args.n_estimators)
