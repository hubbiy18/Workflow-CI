import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

def upload_to_drive(filepath, filename):
    # Load credentials from environment variable
    creds_info = json.loads(os.environ["GDRIVE_CREDENTIALS"])
    creds = service_account.Credentials.from_service_account_info(creds_info)

    # Build Drive service
    service = build("drive", "v3", credentials=creds)

    # Metadata dan file upload
    file_metadata = {"name": filename}
    media = MediaFileUpload(filepath, resumable=True)

    file = service.files().create(body=file_metadata, media_body=media, fields="id").execute()
    print(f"File uploaded to Google Drive with ID: {file.get('id')}")

if __name__ == "__main__":
    import sys
    # Argument ke-1 = path file model
    model_path = sys.argv[1] if len(sys.argv) > 1 else "model.pkl"
    upload_to_drive(model_path, "model.pkl")
