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
import sys
import tempfile # Tambahkan ini

def run_modelling(cleaned_filepath="diabetes_cleaned.csv", model_output="rf_model.pkl", n_estimators=100):
    mlflow.sklearn.autolog(disable=True) # Tetap nonaktifkan autologging

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

        # Simpan model lokal dengan joblib (untuk GDrive upload)
        joblib.dump(model, model_output)
        print(f"\nModel disimpan ke file lokal: {model_output}")

        # --- PERBAIKAN UTAMA DI SINI ---
        # 1. Simpan model dalam format MLflow pyfunc ke direktori sementara
        # 2. Log direktori ini sebagai artifact
        # 3. Register model ke Model Registry

        model_path_for_mlflow = "model_mlflow_format" # Nama direktori di dalam artifacts
        local_temp_model_dir = None # Inisialisasi

        try:
            # Gunakan direktori sementara untuk menyimpan model dalam format MLflow
            # Ini memastikan tidak ada masalah izin atau path relatif
            with tempfile.TemporaryDirectory() as tmpdir:
                local_temp_model_dir = os.path.join(tmpdir, model_path_for_mlflow)
                os.makedirs(local_temp_model_dir, exist_ok=True) # Pastikan direktori ada

                # Gunakan mlflow.pyfunc.save_model untuk menyimpan model
                mlflow.pyfunc.save_model(
                    path=local_temp_model_dir,
                    python_model=mlflow.pyfunc.PythonModel(
                        artifact_path="model", # Ini adalah artifact_path di dalam model_mlflow_dir
                        loader_module=mlflow.sklearn.FLAVOR_NAME, # Menggunakan loader sklearn
                        data_path="model.pkl" # Nama file model di dalam artifact_path model_mlflow_dir
                    ),
                    conda_env={
                        "channels": ["defaults", "conda-forge"],
                        "dependencies": [
                            f"python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                            "pip",
                            {
                                "pip": [
                                    "mlflow",
                                    "scikit-learn",
                                    "pandas",
                                    "imbalanced-learn",
                                    "joblib"
                                ]
                            }
                        ]
                    },
                    # Tambahan: Simpan juga model sklearn sebagai file .pkl di dalam direktori model MLflow
                    artifacts={
                        "model.pkl": joblib.dump(model, os.path.join(local_temp_model_dir, "model.pkl"))
                    }
                )

                print(f"\nModel MLflow format disimpan secara lokal di: {local_temp_model_dir}")
                print(f"Isi dari {local_temp_model_dir}: {os.listdir(local_temp_model_dir)}")

                # Log seluruh direktori yang baru dibuat sebagai artifact
                mlflow.log_artifacts(local_dir=local_temp_model_dir, artifact_path=model_path_for_mlflow)
                print(f"Direktori '{model_path_for_mlflow}' berhasil dicatat sebagai artifact MLflow.")

                # Opsional: Daftarkan model ke Model Registry
                mlflow.register_model(
                    model_uri=f"runs:/{run.info.run_id}/{model_path_for_mlflow}",
                    name="DiabetesPredictionModel"
                )
                print("Model berhasil didaftarkan ke MLflow Model Registry.")

        except Exception as e:
            print(f"\nERROR KRITIS saat menyimpan/mencatat model MLflow format: {e}")
            sys.exit(1)

        # Logging file .pkl yang disimpan manual juga sebagai artifact (untuk GDrive)
        mlflow.log_artifact(model_output)
        print(f"File {model_output} (rf_model.pkl) juga dicatat sebagai artifact.")


        # Debug: tampilkan isi artifact dir utama
        artifacts_dir = os.path.join("mlruns", "0", run.info.run_id, "artifacts")
        print("\nIsi direktori artifact setelah semua logging:")
        try:
            if os.path.exists(artifacts_dir):
                print(os.listdir(artifacts_dir))
                # Verifikasi direktori model_mlflow_format yang baru
                final_model_artifact_path = os.path.join(artifacts_dir, model_path_for_mlflow)
                if os.path.exists(final_model_artifact_path) and os.path.isdir(final_model_artifact_path):
                    print(f"Isi direktori {final_model_artifact_path}:")
                    print(os.listdir(final_model_artifact_path))
                else:
                    print(f"Direktori '{model_path_for_mlflow}' ({final_model_artifact_path}) tidak ditemukan di artifacts utama.")
            else:
                print(f"Direktori artifacts ({artifacts_dir}) tidak ditemukan.")
        except Exception as e:
            print(f"Error saat listing artifacts directory: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()

    run_modelling(n_estimators=args.n_estimators)
