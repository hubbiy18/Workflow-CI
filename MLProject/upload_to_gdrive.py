import sys
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

def upload_to_gdrive(file_path):
    credentials = service_account.Credentials.from_service_account_file(
        "credentials.json",
        scopes=["https://www.googleapis.com/auth/drive.file"]
    )

    service = build("drive", "v3", credentials=credentials)

    file_metadata = {
        "name": os.path.basename(file_path)
    }
    media = MediaFileUpload(file_path, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields="id").execute()

    print(f"File uploaded to Google Drive with ID: {file.get('id')}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python upload_to_gdrive.py <file_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    if not os.path.exists(model_path):
        print(f"Model file not found at: {model_path}")
        sys.exit(1)

    upload_to_gdrive(model_path)
