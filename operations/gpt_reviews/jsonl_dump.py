import concurrent.futures
import io
import json
import os
import sys
import threading
import traceback

import nbformat
from dotenv import find_dotenv, load_dotenv
from fuzzywuzzy import fuzz
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from openai import OpenAI
from utils import get_delivered_df, DATA_DIR, PROJECT_ROOT

from src.llm_reviewer.constants import Roles
from src.llm_reviewer.llm_api import LLMAPIFactory, make_llm_request


def get_closest_match(query, choices):
    """
    Get the closest match(es) to a query string from a list of choices.

    :param query: The query string.
    :param choices: A list of strings to match against.
    :param limit: The maximum number of matches to return.
    """
    best_role = None
    best_score = 0
    for choice in choices:
        score = fuzz.ratio(query, choice)
        if score > best_score and score > 25:
            best_score = score
            best_role = choice

    return best_role, best_score


def count_empty_from_end(cells):
    count = 0
    for message in reversed(cells):
        if message["source"].strip() == "":
            count += 1
        else:
            break
    return count


def extract_messages(notebook):
    """
    Parse a notebook and extract the message objects.

    :param notebook: The notebook object.
    """
    messages = []
    cut_tail = count_empty_from_end(notebook.cells)
    cells = notebook.cells[2:]
    if cut_tail:
        cells = cells[:-cut_tail]
    for cell in cells:
        if cell["cell_type"] == "markdown":
            headers = ["**User**", "**Assistant**"]
        elif cell["cell_type"] == "code":
            headers = ["# User", "# Assistant"]
        else:
            raise Exception(f'Unknown cell type {cell["cell_type"]}')

        lines = cell["source"].split("\n")
        first_line = lines[0]
        role, score = get_closest_match(first_line, headers)
        if score > 50:
            valid_role = role.replace("*", "").replace("#", "").strip()
            content = "\n".join(lines[1:]).strip("\n")
        else:
            valid_role = ""
            content = cell["source"]
        message = {"role": valid_role, "content": content, "type": cell["cell_type"]}
        messages.append(message)

    return messages


def predict_role(messages_subsequence):
    try:
        llm_client = LLMAPIFactory().get()
        missing_role = make_llm_request(
            llm_client,
            model="gpt-4-1106-preview",
            messages=[
                {
                    "role": "system",
                    "content": "Your task is to accurately predict whether the empty role is a User or an Assistant. You are only allowed to reply with a single word: 'User' or 'Assistant'.",
                },
                {
                    "role": "user",
                    "content": f"Here's a part of the conversation including an empty role:\n\n{messages_subsequence}",
                },
            ],
            temperature=0,
        )
        print("Filling out missing header...")
        assert missing_role in ["User", "Assistant"]
        return missing_role, None
    except Exception as e:
        traceback.print_exc()
        return None, e


def fix_missing_roles(messages):
    """
    Fix missing roles in a list of messages.

    :param messages: The list of messages.
    """
    errors = []
    for i in range(len(messages)):
        if messages[i]["role"] == "":
            subsequence = messages[max(0, i - 2) : min(len(messages), i + 3)]
            messages[i]["role"], error = predict_role(subsequence)
            if error is not None:
                errors.append(error)
    return messages, errors


def extract_metadata(notebook):
    # # Extract the first cell
    first_cell = notebook.cells[0]
    lines = first_cell["source"].split("\n")
    metadata = {"topic": "debugging_and_tracing"}
    for line in lines:
        if "**Python Topics**" in line or "**Tecgbucak Topic**" in line:
            metadata["topic"] = line.split(" - ")[1]
        if "**Type**" in line:
            metadata["type"] = line.split(" - ")[1]
        if "**Target Number of Turns (User + Assistant)**" in line:
            metadata["target_turns"] = line.split(" - ")[1]

    return metadata


def notebook_parser(notebook):
    messages = extract_messages(notebook)
    metadata = extract_metadata(notebook)
    messages, errors = fix_missing_roles(messages)
    if errors:
        raise Exception("Failed to predict missing roles.")
    return {"metadata": metadata, "messages": messages}


def download_and_parse_notebook(service_account_file, file_id, revision_id=None):
    # Authenticate with the service account
    credentials = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=["https://www.googleapis.com/auth/drive"]
    )
    service = build("drive", "v3", credentials=credentials)

    # Request to download the file, optionally specifying a revision
    if revision_id is not None:
        request = service.revisions().get_media(fileId=file_id, revisionId=revision_id)
    else:
        request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)

    # Download the file
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print("Download progress: %d%%." % int(status.progress() * 100))

    # Move the buffer's pointer to the beginning
    fh.seek(0)

    # Open the notebook
    notebook = nbformat.read(fh, as_version=4)

    # Try to parse the notebook, return None if an exception occurs
    try:
        parsed_nb = notebook_parser(notebook)
    except Exception as e:
        print(f"Error parsing notebook: {e}")
        return None

    return {
        "id": file_id,
        "colab_link": f"https://colab.research.google.com/drive/{file_id}",
        "revision_id": revision_id,
        **parsed_nb,
    }


service_account_path = PROJECT_ROOT + "creds/google__sa.json"
DELIVERY_SHEET_ID = "1eUif5I8xhHU8fY0X9v8r2JI9hWPh7Dq_9VXpSIHwww4"
destination_folder_url = (
    "https://drive.google.com/drive/folders/1QJByXlbA7UCQMRASsrswtVmUVhMvT1gH"
)

load_dotenv(find_dotenv())


def download_parse_delivered_into_jsonl(
    batch_ids=[1, 2, 3, 4], max_workers=20, no_work=False
):
    delivered_df = get_delivered_df(batch_ids)  # Assuming batch_id 4 is required
    if not no_work:
        parsed_conversations = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            file_ids = [link.split("/")[-1] for link in delivered_df["task_link"]]
            parsed_conversations = list(
                executor.map(
                    download_and_parse_notebook,
                    [service_account_path] * len(file_ids),
                    file_ids,
                )
            )

        for conversation in parsed_conversations:
            if conversation is None:
                continue
            colab_link = conversation["colab_link"]
            batch_id = None
            for b_id, task_link in zip(
                delivered_df["batch_id"], delivered_df["task_link"]
            ):
                if colab_link.endswith(task_link.split("/")[-1]):
                    batch_id = b_id
                    break
            if batch_id is None:
                raise
            batch_name = f"batch_{batch_id}"
            batch_folder = f"{DATA_DIR}jsonl_conversations/{batch_name}/"
            if not os.path.exists(batch_folder):
                os.makedirs(batch_folder)
            drive_id = conversation["id"]
            with open(f"{batch_folder}{drive_id}.json", "w") as f:
                f.write(json.dumps({"batch_id": batch_id, **conversation}))
    else:
        parsed_conversations = []
        for batch_id in batch_ids:
            batch_name = f"batch_{batch_id}"
            batch_folder = f"{DATA_DIR}jsonl_conversations/{batch_name}/"
            if os.path.exists(batch_folder):
                for file_name in os.listdir(batch_folder):
                    if file_name.endswith(".json"):
                        with open(f"{batch_folder}{file_name}", "r") as f:
                            conversation = json.load(f)
                            parsed_conversations.append(conversation)
            else:
                raise Exception(f"Batch folder for {batch_name} not found")

    return {"delivered_df": delivered_df, "conversations": parsed_conversations}


if __name__ == "__main__":
    download_parse_delivered_into_jsonl()
