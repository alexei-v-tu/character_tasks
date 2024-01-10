from typing import List
from googleapiclient.discovery import Resource
from src.gdrive_api.utils import (
    create_folder_path,
    extract_folder_id,
    list_all_files_in_folder,
)


def clone_contents(service: Resource, source_id: str, dest_id: str):
    """Clone the contents of a Google Drive folder to another folder.

    Args:
        service: The Google Drive service resource.
        source_id: The ID of the source folder in Google Drive.
        dest_id: The ID of the destination folder in Google Drive.
    """
    all_files = list_all_files_in_folder(service, source_id)
    for item in all_files:
        if item["mimeType"] == "application/vnd.google-apps.folder":
            # It's a folder, create it and clone its contents
            new_folder_id = create_folder_path(service, item["name"], dest_id)
            clone_contents(service, item["id"], new_folder_id)
        else:
            # It's a file, copy it to the destination folder
            file_metadata = {"parents": [dest_id]}
            service.files().copy(fileId=item["id"], body=file_metadata).execute()
            print(f"Copied file '{item['name']}' to folder ID '{dest_id}'.")


def clone_drive_folder(
    service: Resource,
    source_folder: str,
    destination_folder: str,
    is_url: bool = True,
):
    """Clone a Google Drive folder with all its contents to another folder.

    Args:
        service: The Google Drive service resource.
        source_folder: The ID or URL of the source folder in Google Drive.
        destination_folder: The ID or URL of the destination folder in Google Drive.
        is_url: A flag indicating whether the provided source and destination are URLs. Default is True.
    """
    source_folder_id = extract_folder_id(source_folder, is_url)
    destination_folder_id = extract_folder_id(destination_folder, is_url)
    # Check if source folder exists
    source_folder = service.files().get(fileId=source_folder_id).execute()
    if not source_folder:
        raise ValueError(f"Source folder with ID '{source_folder_id}' does not exist.")

    # Check if destination folder exists
    destination_folder = service.files().get(fileId=destination_folder_id).execute()
    if not destination_folder:
        raise ValueError(
            f"Destination folder with ID '{destination_folder_id}' does not exist."
        )
    # Start the cloning process from the source folder to the destination
    clone_contents(service, source_folder_id, destination_folder_id)
    print(
        f"Cloning of folder ID '{source_folder_id}' to folder ID '{destination_folder_id}' completed."
    )
