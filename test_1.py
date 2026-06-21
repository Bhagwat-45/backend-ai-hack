from services.blob_storage import upload_csv

import os
from dotenv import load_dotenv

load_dotenv()

print("CONNECTION STRING:", os.getenv("AZURE_STORAGE_CONNECTION_STRING"))

blob_path = upload_csv(
    "test.csv",
    "test.csv"
)

print(blob_path)


from services.blob_storage import download_csv_to_dataframe

blob_name = "test.csv"

df = download_csv_to_dataframe(blob_name)

print(df.head())
print(df.columns.tolist())
print(len(df))

from services.blob_storage import download_csv_to_dataframe
from services.case_analysis import top_longest_cases

df = download_csv_to_dataframe(blob_name)

print(top_longest_cases(df))