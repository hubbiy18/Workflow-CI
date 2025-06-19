import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import joblib
import argparse

def run_modelling(cleaned_filepath="diabetes_cleaned.csv", model_output="rf_model.pkl", n_estimators=100):
    # Nonaktifkan autolog agar kita log manual
    mlflow.sklearn.autolog(disable=True)

    # Baca dataset
    df = pd.read_csv(cleaned_filepath)
    df = df.astype({col: 'float64' for col in df.select_dtypes(include='int').columns})

    # Pisahkan fitur dan label
    X = df.drop("diabetes", axis=1)
    y = df["diabetes"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE untuk data imbalance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("Distribusi label setelah SMOTE (training):")
    print(pd.Series(y_train_res).value_counts())

    # Inisialisasi model
    model = RandomForestClassifier(
        random_state=42,
        class_weight='balanced',
        n_estimators=n_estimators
    )

    with mlflow.start_run():
        # Training
        model.fit(X_train_res, y_train_res)

        # Prediksi
        y_pred = model.predict(X_test)
        acc = model.score(X_test, y_test)

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=4))

        # Logging parameter dan metric
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("smote", True)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", acc)

        # Simpan model secara lokal
        joblib.dump(model, model_output)
        print(f"\nModel disimpan ke file lokal: {model_output}")

        # Logging model ke MLflow (gunakan `name=` untuk versi terbaru MLflow)
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",  # menggantikan artifact_path
            input_example=X_test.iloc[:5]
        )

        # Optional: log file .pkl juga ke MLflow artifacts (lokal)
        mlflow.log_artifact(model_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()

    run_modelling(n_estimators=args.n_estimators)
