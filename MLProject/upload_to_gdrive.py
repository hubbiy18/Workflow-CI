from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os

def upload_to_gdrive(file_path, folder_id=None):
    # Load credentials from file (gdrive_credentials.json)
    credentials = service_account.Credentials.from_service_account_file(
        'gdrive_credentials.json',
        scopes=['https://www.googleapis.com/auth/drive']
    )

    service = build('drive', 'v3', credentials=credentials)

    file_metadata = {'name': os.path.basename(file_path)}
    if folder_id:
        file_metadata['parents'] = [folder_id]

    media = MediaFileUpload(file_path, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()

    print(f"File uploaded to Google Drive with ID: {file.get('id')}")

if __name__ == "__main__":
    # Ubah ini ke path sebenarnya model.pkl hasil training
    model_path = "MLProject/mlruns/0/<RUN_ID>/artifacts/model/model.pkl"
    upload_to_gdrive(model_path)
