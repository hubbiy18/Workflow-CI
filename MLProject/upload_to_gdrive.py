import os
import json
import base64
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

def upload_to_gdrive(file_path, folder_id):
    # 1. Ambil credential base64 dari environment variable
    encoded_credentials = os.getenv("GDRIVE_CREDENTIALS")

    if not encoded_credentials:
        raise ValueError("GDRIVE_CREDENTIALS secret not found!")

    # 2. Decode dan simpan ke file sementara
    decoded = base64.b64decode(encoded_credentials)
    with open("gdrive_credentials.json", "wb") as f:
        f.write(decoded)

    # 3. Gunakan file tersebut untuk autentikasi
    credentials = service_account.Credentials.from_service_account_file("gdrive_credentials.json")
    service = build('drive', 'v3', credentials=credentials)

    file_metadata = {
        'name': os.path.basename(file_path),
        'parents': [folder_id]
    }
    media = MediaFileUpload(file_path)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f"File uploaded. ID: {file.get('id')}")

if __name__ == "__main__":
    file_path = "model.pkl"  # Pastikan model.pkl sudah ada
    folder_id = os.getenv("GDRIVE_FOLDER_ID")  # Simpan juga folder_id di GitHub secret (opsional)
    upload_to_gdrive(file_path, folder_id)
