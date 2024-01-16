import re
from typing import Optional
from googleapiclient.discovery import Resource


def extract_file_id(file: str, is_url: bool = True) -> str:
    """Extract the file ID from a Google Drive file URL or ID.

    Args:
        file: The URL or ID of the file in Google Drive.
        is_url: A flag indicating whether the provided file is a URL. Default is True.

    Returns:
        The ID of the file.
    """
    if is_url:
        try:
            file_id = file.split("/d/")[1].split("/")[0]
            if not re.match(r"^[a-zA-Z0-9_-]+$", file_id):
                raise ValueError(
                    f"Invalid file ID: {file_id}. Please provide a valid Google Drive file ID."
                )
        except IndexError:
            raise ValueError(
                f"Invalid file URL: {file}. Please provide a valid Google Drive file URL."
            )
    else:
        if "/" in file:
            raise ValueError(
                f"Invalid file ID: {file}. Please provide a valid Google Drive file ID."
            )
        file_id = file
    return file_id


def extract_folder_id(folder: str, is_url: bool = True) -> str:
    """Extract the folder ID from a Google Drive folder URL or ID.

    Args:
        folder: The URL or ID of the folder in Google Drive.
        is_url: A flag indicating whether the provided folder is a URL. Default is True.

    Returns:
        The ID of the folder.
    """
    if is_url:
        try:
            folder_id = folder.split("/folders/")[1].split("?")[0]
            if not re.match(r"^[a-zA-Z0-9_-]+$", folder_id):
                raise ValueError(
                    f"Invalid folder ID: {folder_id}. Please provide a valid Google Drive folder ID."
                )
        except IndexError:
            raise ValueError(
                f"Invalid folder URL: {folder}. Please provide a valid Google Drive folder URL."
            )
    else:
        if "/" in folder:
            raise ValueError(
                f"Invalid folder ID: {folder}. Please provide a valid Google Drive folder ID."
            )
        folder_id = folder
    return folder_id


def get_nested_folder_id(
    service: Resource, folder_path: str, parent_id: str
) -> Optional[str]:
    """Retrieve the ID of a nested folder in Google Drive using a path.

    Args:
        service: The Google Drive service resource.
        folder_path: The path of the folder to find, may include nested folders.
        parent_id: The ID of the parent folder.

    Returns:
        The ID of the nested folder or None if not found.
    """
    folder_names = folder_path.strip("/").split("/")
    if folder_names == ["."]:
        return parent_id
    for folder_name in folder_names:
        query = f"name = '{folder_name}' and '{parent_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        response = (
            service.files()
            .list(q=query, spaces="drive", fields="files(id, name)")
            .execute()
        )
        folders = response.get("files", [])
        if not folders:
            return None
        # Assuming the first match is the correct one, as folder names can be non-unique
        parent_id = folders[0].get("id")
    return parent_id


def create_folder_path(service: Resource, folder_path: str, parent_id: str) -> str:
    """Create a new folder path in Google Drive, creating subfolders as needed.

    Args:
        service: The Google Drive service resource.
        folder_path: The path of the folder to create, may include nested folders.
        parent_id: The ID of the parent folder.

    Returns:
        The ID of the last subfolder in the path.
    """
    folder_names = folder_path.split("/")
    for folder_name in folder_names:
        folder_id = get_nested_folder_id(service, folder_name, parent_id)
        if folder_id is None:
            file_metadata = {
                "name": folder_name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [parent_id],
            }
            folder = service.files().create(body=file_metadata, fields="id").execute()
            folder_id = folder.get("id")
        parent_id = folder_id
    return parent_id


def get_file_id(service: Resource, file_name: str, parent_id: str) -> Optional[str]:
    """Retrieve the ID of a file in Google Drive.

    Args:
        service: The Google Drive service resource.
        file_name: The name of the file to find.
        parent_id: The ID of the parent folder.

    Returns:
        The ID of the file or None if not found.
    """
    print("Getting file id...")
    query = f"name = '{file_name}' and '{parent_id}' in parents and trashed = false"
    response = (
        service.files()
        .list(q=query, spaces="drive", fields="files(id, name)")
        .execute()
    )
    print("processing response...")
    for file in response.get("files", []):
        if file.get("name") == file_name:
            return file.get("id")
    return None


def list_all_files_in_folder(service: Resource, folder_id: str) -> list[dict]:
    """Get all files in a given Google Drive folder.

    Args:
        service: The Google Drive service resource.
        folder_id: The ID of the folder.

    Returns:
        A list of dictionaries, each representing a file.
    """
    query = f"'{folder_id}' in parents and trashed = false"
    response = (
        service.files()
        .list(
            q=query,
            spaces="drive",
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=None,
        )
        .execute()
    )
    all_files = response.get("files", [])
    page_number = 2
    while "nextPageToken" in response:
        print(f"Processing page number {page_number}...")
        response = (
            service.files()
            .list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType)",
                pageToken=response["nextPageToken"],
            )
            .execute()
        )
        all_files.extend(response.get("files", []))
        page_number += 1
    return all_files


def map_all_gdrive_files_to_ids(
    service: Resource, folder: str, parent_path: str = ".", is_url=True
) -> dict[str, str]:
    """Recursively get all files in a Google Drive folder.

    Args:
        service: The Google Drive service resource.
        folder: The ID or URL of the folder.
        parent_path: The relative path of the parent folder.
        is_url: Whether folder is an ID or URL.

    Returns:
        A dictionary mapping relative file paths to their corresponding file IDs.

    Raises:
        ValueError: If the folder does not exist.
    """
    folder_id = extract_folder_id(folder, is_url)

    all_files = list_all_files_in_folder(service, folder_id)

    files = {}
    for file in all_files:
        file_path = f"{parent_path}/{file.get('name')}"
        if file.get("mimeType") != "application/vnd.google-apps.folder":
            files[file_path] = file.get("id")
        else:
            files.update(
                map_all_gdrive_files_to_ids(
                    service, file.get("id"), file_path, is_url=False
                )
            )

    return files
