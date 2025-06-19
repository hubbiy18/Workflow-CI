import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

def upload_to_gdrive(file_path, folder_id):
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    if not credentials_path or not os.path.exists(credentials_path):
        raise ValueError("GDRIVE_CREDENTIALS secret not found or file missing!")

    credentials = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=["https://www.googleapis.com/auth/drive.file"]
    )

    service = build("drive", "v3", credentials=credentials)

    file_metadata = {
        "name": os.path.basename(file_path),
        "parents": [folder_id]
    }

    media = MediaFileUpload(file_path, resumable=True)

    uploaded = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id"
    ).execute()

    print(f"File uploaded successfully. File ID: {uploaded.get('id')}")

if __name__ == "__main__":
    # Path relatif karena file ada di MLProject/
    file_path = os.path.abspath("model.pkl")

    # Ganti folder_id dengan folder Google Drive kamu
    folder_id = "1tE5eCxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # <- Ganti dengan ID folder "MLflow_Models_Febie"

    upload_to_gdrive(file_path, folder_id)
