import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    mlflow.set_experiment("Diabetes Prediction")

    with mlflow.start_run():
        df = pd.read_csv("diabetes_cleaned.csv")

        X = df.drop("diabetes", axis=1)
        y = df["diabetes"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Log metrics and model
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")  # Ini akan buat artifacts/model/

        print("Model trained and logged to MLflow.")

if __name__ == "__main__":
    main()
