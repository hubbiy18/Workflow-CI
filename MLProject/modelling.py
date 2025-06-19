import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import joblib
import argparse
import os

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

    model = RandomForestClassifier(
        random_state=42,
        class_weight='balanced',
        n_estimators=n_estimators
    )

    with mlflow.start_run() as run:
        model.fit(X_train_res, y_train_res)

        y_pred = model.predict(X_test)
        acc = model.score(X_test, y_test)

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=4))

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("smote", True)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", acc)

        # Simpan model lokal (ini akan menyimpan rf_model.pkl di direktori kerja MLProject)
        joblib.dump(model, model_output)
        print(f"\nModel disimpan ke file lokal: {model_output}")

        # Logging model dalam format MLflow (wajib artifact_path='model')
        # Pastikan ini dijalankan dengan benar
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",  # ini akan membuat mlruns/.../artifacts/model/
                input_example=X_test.iloc[:5]
            )
            print("\nmlflow.sklearn.log_model berhasil mencatat model.")
        except Exception as e:
            print(f"\nERROR: mlflow.sklearn.log_model GAGAL: {e}")
            # Anda bisa menambahkan sys.exit(1) di sini jika ingin pipeline gagal total
            # sys.exit(1)

        # Logging file .pkl juga sebagai artifact
        mlflow.log_artifact(model_output)
        print(f"File {model_output} juga dicatat sebagai artifact.")


        # Debug: tampilkan isi artifact dir
        artifacts_dir = os.path.join("mlruns", "0", run.info.run_id, "artifacts")
        print("\nIsi direktori artifact setelah semua logging:")
        try:
            if os.path.exists(artifacts_dir):
                print(os.listdir(artifacts_dir))
                # Tambahan: Periksa isi direktori 'model' jika ada
                model_artifact_path = os.path.join(artifacts_dir, "model")
                if os.path.exists(model_artifact_path) and os.path.isdir(model_artifact_path):
                    print(f"Isi direktori {model_artifact_path}:")
                    print(os.listdir(model_artifact_path))
                else:
                    print(f"Direktori 'model' ({model_artifact_path}) tidak ditemukan di artifacts.")
            else:
                print(f"Direktori artifacts ({artifacts_dir}) tidak ditemukan.")
        except Exception as e:
            print(f"Error saat listing artifacts directory: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()

    run_modelling(n_estimators=args.n_estimators)
