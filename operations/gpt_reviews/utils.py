import os
import sys
import json

PROJECT_ROOT = os.getcwd()

for _ in range(5):
    if os.path.isfile(os.path.join(PROJECT_ROOT, ".env")):
        break
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
else:
    raise FileNotFoundError(
        "Could not find .env file within 5 levels of the current directory."
    )

sys.path.append(PROJECT_ROOT)

PROJECT_ROOT += "/"
DATA_DIR = PROJECT_ROOT + "data/"


service_account_path = PROJECT_ROOT + "creds/google__sa.json"
tracking_sheet_id = "1qBU7Kvuuij2fxbqPxebReKMxWgIBmOIE5Gi4ZuX0j_4"
delivery_sheet_id = "1eUif5I8xhHU8fY0X9v8r2JI9hWPh7Dq_9VXpSIHwww4"


import pandas as pd

from src.sheets_utils import download_sheet_as_df

turing_palette = [
    "#326FDC",  # Celtic Blue
    "#47ABFD",  # Argentinian Blue
    "#959595",  # Battleship gray
    "#FFFFFF",  # White
    "#EFEFEF",  # Anti-flash white
    "#000000",  # Black
]


def process_batch(batch_folder, limit=None):
    file_list = [file for file in os.listdir(batch_folder) if file.endswith(".json")]
    if limit is not None:
        file_list = file_list[:limit]
    conversations = []
    for file_name in file_list:
        with open(f"{batch_folder}/{file_name}", "r") as f:
            conversation = json.load(f)
            conversations.append(conversation)
    return conversations


def get_delivered_df(batch_ids=[1, 2, 3, 4]):
    delivered_dfs = []
    for batch_id in batch_ids:
        # if batch_id == 5:
        #    delivered_b5_mock_df = pd.read_csv(DATA_DIR + "batch5_delivered_mock.csv")
        #    df = delivered_b5_mock_df
        # else:
        df = download_sheet_as_df(
            service_account_path, delivery_sheet_id, f"Batch {batch_id}"
        )
        df = df.assign(batch_id=batch_id)
        delivered_dfs.append(df)
    delivered_df = pd.concat(delivered_dfs, ignore_index=True)
    return delivered_df
