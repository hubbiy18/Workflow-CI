import sys
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

def upload_to_gdrive(file_path):
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    SERVICE_ACCOUNT_FILE = 'credentials.json'

    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )

    service = build('drive', 'v3', credentials=credentials)

    file_metadata = {'name': os.path.basename(file_path)}
    media = MediaFileUpload(file_path, resumable=True)
    
    file = service.files().create(
        body=file_metadata, media_body=media, fields='id'
    ).execute()

    print(f"Uploaded file ID: {file.get('id')}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python upload_to_gdrive.py path/to/model.pkl")
        sys.exit(1)

    model_path = sys.argv[1]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    upload_to_gdrive(model_path)
