from google.oauth2 import service_account
from googleapiclient.discovery import build
import logging

logging.basicConfig(level=logging.DEBUG)

SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = "/app/client.json"

try:
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    service = build(
        'drive', 'v3',
        credentials=credentials,
        static_discovery=False,  # ここを追加
        cache_discovery=False
    )
    logging.info("Drive API service created successfully: %s", service)
except Exception as e:
    logging.exception("Error creating Drive API service:")
