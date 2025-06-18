name: CI MLflow Train

on:
  push:
    paths:
      - 'MLProject/**'
  workflow_dispatch:

jobs:
  train:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v3

      - name: Set up Miniconda
        uses: conda-incubator/setup-miniconda@v2
        with:
          auto-update-conda: true
          environment-file: MLProject/conda.yaml
          activate-environment: mlflow-env
          auto-activate-base: false

      - name: Install Python dependencies
        run: |
          pip install mlflow scikit-learn imbalanced-learn pandas joblib \
                      google-api-python-client google-auth google-auth-httplib2 google-auth-oauthlib

      - name: Run MLflow Project
        run: |
          cd MLProject
          mlflow run . --env-manager=local

      - name: Get latest MLflow run_id
        id: get_run_id
        run: |
          RUN_ID=$(ls -td MLProject/mlruns/0/* | head -1 | xargs -n 1 basename)
          echo "RUN_ID=$RUN_ID" >> $GITHUB_ENV

      - name: Check model artifact existence
        run: |
          MODEL_PATH="MLProject/mlruns/0/${{ env.RUN_ID }}/artifacts/model/model.pkl"
          if [ ! -f "$MODEL_PATH" ]; then
            echo "Model artifact not found at $MODEL_PATH"
            exit 1
          fi

      - name: Save GDrive credentials
        run: |
          echo '${{ secrets.GDRIVE_CREDENTIALS }}' > credentials.json

      - name: Upload model to Google Drive
        run: |
          python MLProject/upload_to_gdrive.py "MLProject/mlruns/0/${{ env.RUN_ID }}/artifacts/model/model.pkl"

      - name: Log in to Docker Hub
        run: echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin

      - name: Build Docker Image from MLflow Model
        run: |
          mlflow models build-docker -m "MLProject/mlruns/0/${{ env.RUN_ID }}/artifacts/model" -n ${{ secrets.DOCKER_USERNAME }}/mlflow-model:latest

      - name: Push Docker Image
        run: |
          docker push ${{ secrets.DOCKER_USERNAME }}/mlflow-model:latest
