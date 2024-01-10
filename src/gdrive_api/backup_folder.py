from typing import List

from googleapiclient.discovery import Resource

from src.gdrive_api.utils import create_folder_path, extract_folder_id
from src.gdrive_api.folder_clone import clone_drive_folder


def backup_folder(
    service: Resource,
    source_folder: str,
    destination_parent: str,
    subfolder_name: str,
    is_url: bool = True,
):
    """Backup a Google Drive folder to a subfolder in another folder.

    Args:
        service: The Google Drive service resource.
        source_folder: The ID or URL of the source folder in Google Drive.
        destination_parent: The ID or URL of the parent folder in Google Drive where the backup will be created.
        subfolder_name: The name of the subfolder to be created in the destination folder.
        is_url: A flag indicating whether the provided source and destination are URLs. Default is True.
    """
    # Extract the folder IDs
    source_folder_id = extract_folder_id(source_folder, is_url)
    destination_parent_id = extract_folder_id(destination_parent, is_url)

    # Create a new subfolder in the destination folder
    subfolder_id = create_folder_path(service, subfolder_name, destination_parent_id)

    # Clone the source folder to the new subfolder
    clone_drive_folder(service, source_folder_id, subfolder_id, is_url=False)
    print(
        f"Backup of folder '{source_folder}' to subfolder '{subfolder_name}' in folder '{destination_parent}' completed."
    )
