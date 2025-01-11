import os
import json

from typing import Dict, Any


def check_directory_path_existence(directory_path: str) -> str:
    """Creates the directory path.

    Creates the absolute path for the directory path given in argument if it does not already exist.

    Args:
        directory_path: A string for the directory path that needs to be created if it does not already exist.

    Returns:
        A string for the absolute directory path.
    """
    # Asserts type of arguments.
    assert isinstance(
        directory_path, str
    ), "Variable directory_path should be of type 'str'."

    # Creates the following directory path if it does not exist.
    home_directory_path = os.getcwd()
    absolute_directory_path = os.path.join(home_directory_path, directory_path)
    if not os.path.isdir(absolute_directory_path):
        os.makedirs(absolute_directory_path)
    return absolute_directory_path


def save_json_file(
    dictionary: Dict[Any, Any], file_name: str, directory_path: str
) -> None:
    """Saves dictionary as a JSON file.

    Converts a dictionary into a JSON file and saves it for future use.

    Args:
        dictionary: A dictionary which needs to be saved.
        file_name: A string for the name with which the file has to be saved.
        directory_path: A string for the path where the file needs to be saved.

    Returns:
        None.
    """
    # Types checks arguments.
    assert isinstance(file_name, str), "Variable file_name should be of type 'str'."
    assert isinstance(
        directory_path, str
    ), "Variable directory_path should be of type 'str'."

    # Checks if the following path exists.
    directory_path = check_directory_path_existence(directory_path)

    # Saves the dictionary or list as a JSON file at the file path location.
    file_path = f"{directory_path}/{file_name}.json"
    with open(file_path, "w") as out_file:
        json.dump(dictionary, out_file, indent=4)
    out_file.close()
    print(f"{file_name}.json file saved successfully.")


def load_text_file(file_name: str, directory_path: str) -> str:
    """Loads text file as a string.

    Loads text file as a string.

    Args:
        file_name: A string for the name of the file that needs to be loaded.
        directory_path: A string for the location where the file needs to be loaded is present.

    Returns:
        A string for the text from the loaded file.

    Exception:
        FileNotFoundError: If the file path does not exist, then this error occurs.
    """
    # Checks type of input documents.
    assert isinstance(file_name, str), "Variable file_name should be of type 'str'."
    assert isinstance(
        directory_path, str
    ), "Variable directory_path should be of type 'str'."

    file_path = f"{directory_path}/{file_name}"

    # Loads the text file as string from the file location.
    try:
        with open(file_path, "r", encoding="utf-8") as out_file:
            text = out_file.read()
        out_file.close()
        return text

    except FileNotFoundError:
        raise FileNotFoundError(f"{file_path} does not exist.")


def save_text_file(text: str, file_name: str, directory_path: str) -> str:
    """Saves string as text file.

    Saves string as text file.

    Args:
        text: A string for the text that needs to be saved.
        file_name: A string for the name of the file that needs to be saved.
        directory_path: A string for the location where the file needs to be saved.

    Returns:
        A string for the text from the loaded file.
    """
    # Checks type of input documents.
    assert isinstance(text, str), "Variable text should be of type 'str'."
    assert isinstance(file_name, str), "Variable file_name should be of type 'str'."
    assert isinstance(
        directory_path, str
    ), "Variable directory_path should be of type 'str'."

    # Checks if the following path exists.
    directory_path = check_directory_path_existence(directory_path)

    # Saves the dictionary or list as a JSON file at the file path location.
    file_path = f"{directory_path}/{file_name}"
    with open(file_path, "w") as out_file:
        out_file.write(text)
    out_file.close()
    print(f"{file_name} file saved successfully.")
