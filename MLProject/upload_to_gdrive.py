from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os

def upload_to_gdrive(file_path, folder_id=None):
    # Cek file ada atau tidak
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

    # Load credentials
    credentials = service_account.Credentials.from_service_account_file(
        'gdrive_credentials.json',
        scopes=['https://www.googleapis.com/auth/drive']
    )

    # Inisialisasi service Google Drive API
    service = build('drive', 'v3', credentials=credentials)

    # Metadata file
    file_metadata = {'name': os.path.basename(file_path)}
    if folder_id:
        file_metadata['parents'] = [folder_id]

    # Upload file
    media = MediaFileUpload(file_path, resumable=True)
    uploaded = service.files().create(body=file_metadata, media_body=media, fields='id').execute()

    print(f"File berhasil diupload ke Google Drive! ID: {uploaded.get('id')}")

if __name__ == "__main__":
    # Ganti path di bawah sesuai lokasi file model.pkl hasil training
    file_path = "model.pkl"  # atau "MLProject/model.pkl" jika path relatif

    # Jika ingin upload ke folder tertentu di Google Drive, masukkan folder_id di sini:
    folder_id = None  # Ganti misalnya: "1A2B3C4D5E6F..."

    upload_to_gdrive(file_path, folder_id)
