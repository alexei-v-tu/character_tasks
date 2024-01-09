import os
import sys


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


def get_delivered_df(batch_ids=[1, 2, 3, 4]):
    delivered_dfs = []
    for batch_id in batch_ids:
        df = download_sheet_as_df(
            service_account_path, delivery_sheet_id, f"Batch {batch_id}"
        )
        df = df.assign(batch_id=batch_id)
        delivered_dfs.append(df)
    delivered_df = pd.concat(delivered_dfs, ignore_index=True)
    return delivered_df
