from azure.storage.blob import BlobServiceClient
import os
from io import BytesIO
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "eventlogs"

blob_service = BlobServiceClient.from_connection_string(
    CONNECTION_STRING
)

def upload_csv(local_path: str, blob_name: str) -> str:

    blob_client = blob_service.get_blob_client(
        container=CONTAINER_NAME,
        blob=blob_name
    )

    with open(local_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    return blob_name




def download_csv_to_dataframe(blob_name: str) -> pd.DataFrame:

    blob_client = blob_service.get_blob_client(
        container=CONTAINER_NAME,
        blob=blob_name
    )

    data = blob_client.download_blob().readall()

    return pd.read_csv(BytesIO(data))