import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import joblib
import os

def run_modelling(cleaned_filepath="diabetes_cleaned.csv", model_output="rf_model.pkl"):
    # Nonaktifkan autolog otomatis
    mlflow.sklearn.autolog(disable=True)

    # === Load Dataset ===
    df = pd.read_csv(cleaned_filepath)
    df = df.astype({col: 'float64' for col in df.select_dtypes(include='int').columns})

    # === Split Feature & Target ===
    X = df.drop("diabetes", axis=1)
    y = df["diabetes"]

    # === Train Test Split ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # === SMOTE Oversampling ===
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("Distribusi label setelah SMOTE (training):")
    print(pd.Series(y_train_res).value_counts())

    # === Model ===
    model = RandomForestClassifier(random_state=42, class_weight='balanced')

    # === Mulai Run MLflow ===
    with mlflow.start_run():
        model.fit(X_train_res, y_train_res)

        y_pred = model.predict(X_test)

        # === Evaluasi ===
        acc = model.score(X_test, y_test)
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=4))

        # === Logging Manual ke MLflow ===
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("smote", True)
        mlflow.log_metric("accuracy", acc)

        # Simpan Model Lokal
        joblib.dump(model, model_output)
        print(f"\nModel disimpan ke: {model_output}")

        # Logging ke MLflow
        input_example = X_test.iloc[:5]
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned_filepath", type=str, default="diabetes_cleaned.csv")
    parser.add_argument("--model_output", type=str, default="rf_model.pkl")
    args = parser.parse_args()

    run_modelling(
        cleaned_filepath=args.cleaned_filepath,
        model_output=args.model_output
    )
