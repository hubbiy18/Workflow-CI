import os
import base64
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

def upload_to_gdrive(file_path, folder_id):
    encoded_credentials = os.getenv("GDRIVE_CREDENTIALS")
    if not encoded_credentials:
        raise ValueError("GDRIVE_CREDENTIALS secret not found!")

    # Simpan hasil decode base64 ke file sementara
    with open("gdrive_credentials.json", "wb") as f:
        f.write(base64.b64decode(encoded_credentials))

    credentials = service_account.Credentials.from_service_account_file(
        "gdrive_credentials.json",
        scopes=["https://www.googleapis.com/auth/drive"]
    )

    service = build("drive", "v3", credentials=credentials)

    file_metadata = {
        "name": os.path.basename(file_path),
        "parents": [folder_id]
    }
    media = MediaFileUpload(file_path, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields="id").execute()
    print(f"File uploaded, ID: {file.get('id')}")

if __name__ == "__main__":
    file_path = "model.pkl"
    folder_id = "GANTI_DENGAN_FOLDER_ID_KAMU"
    upload_to_gdrive(file_path, folder_id)
