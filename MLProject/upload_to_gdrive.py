import sys
import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

def upload_to_gdrive(file_path, folder_id):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} tidak ditemukan.")

    # Path kredensial dari environment variable
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path or not os.path.exists(credentials_path):
        raise ValueError("File kredensial tidak ditemukan atau GOOGLE_APPLICATION_CREDENTIALS belum di-set.")

    # Autentikasi dengan Google Service Account
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    service = build('drive', 'v3', credentials=credentials)

    # Siapkan file upload
    file_metadata = {
        'name': os.path.basename(file_path),
        'parents': [folder_id]
    }
    media = MediaFileUpload(file_path, resumable=True)

    # Upload file
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()

    print(f"File berhasil di-upload ke Google Drive, file ID: {file.get('id')}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python upload_to_gdrive.py <file_path> <folder_id>")
        sys.exit(1)

    file_path = sys.argv[1]
    folder_id = sys.argv[2]

    upload_to_gdrive(file_path, folder_id)
