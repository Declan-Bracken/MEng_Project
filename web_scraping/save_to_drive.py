from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
import os

class GoogleDriveManager:
    def __init__(self, credentials_file='/Users/declanbracken/Development/UofT_Projects/Meng_Project/code_base/client_secret_809384080547-4sfr7l9u8a618keak7b11qan4o63nvh3.apps.googleusercontent.com.json', token_file='token.json', scopes=None):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.scopes = scopes or ['https://www.googleapis.com/auth/drive']
        self.service = self.authenticate_google_drive()

    def authenticate_google_drive(self):
        """Authenticate and return a Google Drive service object."""
        creds = None
        if os.path.exists(self.token_file):
            creds = Credentials.from_authorized_user_file(self.token_file, self.scopes)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_file, self.scopes)
                creds = flow.run_local_server(port=0)
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())
        return build('drive', 'v3', credentials=creds)

    def upload_file(self, file_path, folder_id=None):
        """Upload a file to Google Drive."""
        file_metadata = {'name': os.path.basename(file_path), 'parents': [folder_id] if folder_id else []}
        media = MediaFileUpload(file_path, mimetype='image/jpeg')
        file = self.service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        print(f"File ID: {file['id']} uploaded successfully")
        return file['id']

# Example usage:
if __name__ == "__main__":
    gdrive_manager = GoogleDriveManager()
    folder_key = "14zyq0BXTYrYj81bGlKtYpEG-KL59oNnM"
    # Assuming you have an image called 'example.jpg' and a folder ID 'your_folder_id'
    file_id = gdrive_manager.upload_file('/Users/declanbracken/Development/UofT_Projects/Meng_Project/Transcripts/Web_Scraped_Transcripts_v2/image_3.jpg', folder_key)
    print(f"Uploaded file ID: {file_id}")
