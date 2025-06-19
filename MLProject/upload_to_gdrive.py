import base64
import json
import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account

def upload_to_gdrive(file_path, folder_id):
    encoded_credentials = os.getenv("GDRIVE_CREDENTIALS")
    if not encoded_credentials:
        raise ValueError("GDRIVE_CREDENTIALS secret not found!")

    # Decode base64 string and save it temporarily
    decoded = base64.b64decode(encoded_credentials)
    with open("temp_gdrive_creds.json", "wb") as f:
        f.write(decoded)

    credentials = service_account.Credentials.from_service_account_file(
        "temp_gdrive_creds.json",
        scopes=["https://www.googleapis.com/auth/drive"]
    )

    service = build("drive", "v3", credentials=credentials)
    file_metadata = {
        "name": os.path.basename(file_path),
        "parents": [folder_id]
    }
    media = MediaFileUpload(file_path, resumable=True)
    service.files().create(body=file_metadata, media_body=media, fields="id").execute()
    print(f"File {file_path} uploaded successfully.")

if __name__ == "__main__":
    file_path = "model.pkl"
    folder_id = "YOUR_FOLDER_ID"  # Ganti dengan folder ID Drive kamu
    upload_to_gdrive(file_path, folder_id)
