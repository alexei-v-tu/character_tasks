import os
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Optional

from googleapiclient.discovery import Resource
from googleapiclient.http import MediaFileUpload
from src.gdrive_api.utils import (
    create_folder_path,
    extract_folder_id,
    get_file_id,
    get_nested_folder_id,
)
from src.gdrive_api.auth import build_service


class FolderNotFoundError(Exception):
    """Exception raised when the local source folder is not found."""

    pass


class UploadError(Exception):
    """Exception raised for errors that occur during file upload."""

    pass


def upload_file(
    service: Resource, file_path: str, parent_id: str, force_replace: bool = False
) -> Optional[str]:
    """Upload a file to Google Drive, optionally forcing replacement of existing files.

    Args:
        service: The Google Drive service resource.
        file_path: The path to the file to upload.
        parent_id: The ID of the parent folder in Google Drive.
        force_replace: If True, replace the file if it already exists.

    Returns:
        File url if the file was uploaded, None otherwise.
    """
    file_name = os.path.basename(file_path)
    file_metadata = {"name": file_name, "parents": [parent_id]}
    media = MediaFileUpload(file_path, resumable=True)
    file_id = get_file_id(service, file_name, parent_id)

    if file_id and not force_replace:
        print(f"File '{file_name}' already exists and won't be replaced.")
        return None

    file_url = None
    response = None

    if file_id and force_replace:
        print(f"Replacing existing file '{file_name}' with the new version.")
        response = service.files().update(fileId=file_id, media_body=media).execute()
        print(f"File '{file_name}' has been replaced.")
    else:
        print(f"Uploading new file '{file_name}'.")
        response = (
            service.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )
        print(f"File '{file_name}' has been uploaded.")

    if response:
        file_url = f"https://drive.google.com/uc?id={response['id']}"
        print(f"Uploaded '{file_name}' to folder ID '{parent_id}'.")

    return file_url


def sync_folder_structure(
    service: Resource,
    source_folder_path: str,
    destination_folder_id: str,
) -> None:
    """Create folder structure in Google Drive.

    Args:
        service: The Google Drive service resource.
        source_folder_path: The path to the local folder to upload.
        destination_folder_id: The ID of the destination folder in Google Drive.
    """
    total_dirs = 0
    for root, dirs, _ in os.walk(source_folder_path):
        relative_path = os.path.relpath(root, source_folder_path)
        total_dirs += len(dirs)
    processed_dirs = -1

    for root, dirs, _ in os.walk(source_folder_path):
        relative_path = os.path.relpath(root, source_folder_path)
        current_folder_id = (
            destination_folder_id
            if relative_path == "."
            else get_nested_folder_id(service, relative_path, destination_folder_id)
        )

        if current_folder_id is None:
            create_folder_path(service, relative_path, destination_folder_id)

        processed_dirs += 1
        print(
            f"Synced directory structure: {processed_dirs} out of {total_dirs} directories."
        )


def add_files_to_queue(
    service: Resource,
    source_folder_path: str,
    destination_folder_id: str,
    file_queue: Queue,
) -> None:
    """Add files to be uploaded to the queue.

    Args:
        service: The Google Drive service resource.
        source_folder_path: The path to the local folder to upload.
        destination_folder_id: The ID of the destination folder in Google Drive.
        file_queue: The queue to which files will be added.
    """
    for root, _, files in os.walk(source_folder_path):
        relative_path = os.path.relpath(root, source_folder_path)
        current_folder_id = get_nested_folder_id(
            service, relative_path, destination_folder_id
        )

        for file_name in files:
            file_path = os.path.join(root, file_name)
            relative_file_path = os.path.relpath(file_path, source_folder_path)
            file_queue.put((file_path, current_folder_id, relative_file_path))


def worker(
    creds_file_path: str,
    file_queue: Queue,
    uploaded_files: dict[str, Optional[str]],
    force_replace: bool,
    total_files: int,
) -> None:
    """Function to be run by each thread.

    Args:
        service: The Google Drive service resource.
        file_queue: The queue from which files will be uploaded.
        uploaded_files: Dict of relative file path -> URL for the file after upload, URL is None if it was skipped due to force replace. "ERROR" if there was an error during the upload.
        force_replace: If True, re-upload files even if they exist.
    """
    service = build_service(creds_file_path)
    while not file_queue.empty():
        file_path, current_folder_id, relative_file_path = file_queue.get()
        print(
            f"Processing {relative_file_path} {file_queue.qsize()} left after this one"
        )
        try:
            file_url = upload_file(service, file_path, current_folder_id, force_replace)
            if file_url is not None:
                uploaded_files[relative_file_path] = file_url
            else:
                uploaded_files[relative_file_path] = None

            files_processed = total_files - file_queue.qsize()
            print(f"Files processed: {files_processed}/{total_files}")
        except Exception as e:
            msg = f"An error occurred while uploading '{relative_file_path}': {e}"
            print(msg)
            uploaded_files[relative_file_path] = "ERROR"
        finally:
            file_queue.task_done()


def upload_folder(
    creds_file_path: str,
    source_folder_path: str,
    destination_folder: str,
    force_replace: bool = False,
    is_url: bool = True,
    max_threads: int = 20,
) -> dict[str, Optional[str]]:
    """Recursively upload a local folder to Google Drive.

    Args:
        service: The Google Drive service resource.
        source_folder_path: The path to the local folder to upload.
        destination_folder: The ID or URL of the destination folder in Google Drive.
        force_replace: If True, re-upload files even if they exist.
        is_url: A flag indicating whether the provided destination is a URL. Default is True.
        max_threads: Maximum number of threads to be used for file upload - same as number of files being uploaded concurrently.

    Raises:
        FolderNotFoundError: If the local folder does not exist.
        UploadError: If an error occurs during file upload.

    Returns:
        Dict of relative file path -> URL for the file after upload, URL is None if it was skipped due to force replace. "ERROR" if there was an error during the upload.
    """
    service = build_service(creds_file_path)
    destination_folder_id = extract_folder_id(destination_folder, is_url)
    if not os.path.exists(source_folder_path):
        raise FolderNotFoundError(
            f"Local folder '{source_folder_path}' does not exist."
        )
    sync_folder_structure(service, source_folder_path, destination_folder_id)

    total_files = sum([len(files) for _, _, files in os.walk(source_folder_path)])
    file_queue = Queue()
    uploaded_files = {}

    add_files_to_queue(service, source_folder_path, destination_folder_id, file_queue)

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        for _ in range(min(total_files, max_threads)):
            executor.submit(
                worker,
                creds_file_path,
                file_queue,
                uploaded_files,
                force_replace,
                total_files,
            )
    file_queue.join()

    uploaded_files_count = len(
        [url for url in uploaded_files.values() if url is not None]
    )
    skipped_files_count = total_files - uploaded_files_count

    print(f"Successfully uploaded {uploaded_files_count} files out of {total_files}.")
    print(f"Skipped {skipped_files_count} files.")
    return uploaded_files
