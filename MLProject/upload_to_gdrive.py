import sys
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

def upload_to_gdrive(file_path, folder_id):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

    # Pastikan variabel environment tersedia
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path or not os.path.exists(creds_path):
        raise ValueError("File kredensial Google tidak ditemukan atau GOOGLE_APPLICATION_CREDENTIALS belum di-set!")

    # Autentikasi
    creds = service_account.Credentials.from_service_account_file(
        creds_path,
        scopes=["https://www.googleapis.com/auth/drive.file"]
    )
    service = build("drive", "v3", credentials=creds)

    file_metadata = {
        "name": os.path.basename(file_path),
        "parents": [folder_id]
    }
    media = MediaFileUpload(file_path, resumable=True)

    uploaded_file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id, name"
    ).execute()

    print(f"Berhasil upload file: {uploaded_file.get('name')} (ID: {uploaded_file.get('id')})")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python upload_to_gdrive.py <file_path> <folder_id>")
        sys.exit(1)

    file_path = sys.argv[1]
    folder_id = sys.argv[2]

    upload_to_gdrive(file_path, folder_id)
