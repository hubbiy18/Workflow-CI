import sys
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

FOLDER_ID = "your_google_drive_folder_id"  # ← ganti di sini
SCOPES = ['https://www.googleapis.com/auth/drive.file']
CREDENTIALS_FILE = 'credentials.json'

def upload_to_drive(file_path, file_name):
    creds = service_account.Credentials.from_service_account_file(
        CREDENTIALS_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)

    file_metadata = {'name': file_name, 'parents': [FOLDER_ID]}
    media = MediaFileUpload(file_path, mimetype='application/octet-stream')

    uploaded_file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()

    print("Uploaded to Google Drive with ID:", uploaded_file.get('id'))

if __name__ == '__main__':
    run_id = sys.argv[1]
    model_path = f"MLProject/mlruns/0/{run_id}/artifacts/model/model.pkl"
    upload_to_drive(model_path, "model.pkl")
