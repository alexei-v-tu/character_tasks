from enum import Enum
from googleapiclient.errors import HttpError
from src.gdrive_api.utils import extract_file_id


class Role(Enum):
    VIEWER = "reader"
    EDITOR = "writer"
    REMOVE = "remove"


def remove_permissions(
    service, file: str, user_email: str, is_url: bool = True
) -> bool:
    """
    Remove permissions for a specific user on a specific file.

    :param service: Authorized Google Drive service instance.
    :param file: The ID or URL of the file.
    :param user_email: Email of the user to remove permissions for.
    :param is_url: A flag indicating whether the provided file is a URL. Default is True.
    :return: True if permissions were found and removed, False otherwise.
    """
    file_id = extract_file_id(file, is_url)
    permissions = (
        service.permissions()
        .list(fileId=file_id, fields="permissions(id,emailAddress)")
        .execute()
    )
    for p in permissions["permissions"]:
        if p["emailAddress"] == user_email:
            service.permissions().delete(fileId=file_id, permissionId=p["id"]).execute()
            print(f"Removed {user_email}'s permissions.")
            return True
    print(f"No permissions found for {user_email}.")
    return False


def update_file_permissions(
    service, file: str, user_email: str, role: Role, is_url: bool = True
):
    """
    Update permissions for a specific user on a specific file.

    :param service: Authorized Google Drive service instance.
    :param file: The ID or URL of the file.
    :param user_email: Email of the user to update permissions for.
    :param role: Role to assign to the user.
    :param is_url: A flag indicating whether the provided file is a URL. Default is True.
    """
    if role not in list(Role):
        raise ValueError(f"Invalid role. Must be one of {list(Role)}")
    try:
        file_id = extract_file_id(file, is_url)
        permission = {"type": "user", "role": role.value, "emailAddress": user_email}
        remove_permissions(service, file_id, user_email, is_url=False)
        if role != Role.REMOVE:
            service.permissions().create(fileId=file_id, body=permission).execute()
            print(f"Updated {user_email}'s permissions to {role.value}.")
    except HttpError as error:
        print(f"An error occurred: {error}")


def update_permissions_for_multiple_users(
    service, users_permissions: dict[str, dict[str, Role]], is_url=True
):
    """
    Update permissions for multiple users across multiple files.

    :param service: Authorized Google Drive service instance.
    :param users_permissions: A dictionary mapping user emails to another dictionary that maps file IDs or URLs to Roles.
    :param is_url: A flag indicating whether the provided file is a URL. Default is True.
    """
    total_users = len(users_permissions)
    for user_index, (user_email, files_permissions) in enumerate(
        users_permissions.items(), start=1
    ):
        print(f"Updating user {user_index} of {total_users}: {user_email}")
        total_files = len(files_permissions)
        for file_index, (file_id_or_url, role) in enumerate(
            files_permissions.items(), start=1
        ):
            print(f"Updating file {file_index} of {total_files} for user {user_email}")
            update_file_permissions(service, file_id_or_url, user_email, role, is_url)


def update_permissions_for_multiple_files(
    service, user_email: str, files_permissions: dict[str, Role], is_url=True
):
    """
    Update permissions for a single user across multiple files.

    :param service: Authorized Google Drive service instance.
    :param user_email: Email of the user to update permissions for.
    :param files_permissions: A dictionary mapping file IDs or URLs to Roles.
    :param is_url: A flag indicating whether the provided file is a URL. Default is True.
    """
    update_permissions_for_multiple_users(
        service, {user_email: files_permissions}, is_url
    )


def update_permissions_for_user(
    service, user_email: str, role: Role, file_ids_or_urls: list[str], is_url=True
):
    """
    Update permissions for a single user and a single role across multiple files.

    :param service: Authorized Google Drive service instance.
    :param user_email: Email of the user to update permissions for.
    :param role: Role to assign to the user.
    :param file_ids_or_urls: List of file IDs or URLs to update permissions for.
    :param is_url: A flag indicating whether the provided file is a URL. Default is True.
    """
    files_permissions = {file_id_or_url: role for file_id_or_url in file_ids_or_urls}
    update_permissions_for_multiple_files(
        service, user_email, files_permissions, is_url
    )
