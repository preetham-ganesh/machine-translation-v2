import os
import sys
import warnings
import argparse
import time
import requests
import logging


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.FATAL)


import tensorflow_datasets as tfds

from src.utils import check_directory_path_existence


def download_europarl_dataset(language: str) -> None:
    """Downloads the Europarl dataset for the language given as input by user.

    Downloads the Europarl dataset for the language given as input by user.

    Args:
        language: A string for the language the Europarl dataset should be downloaded.

    Returns:
        None.
    """
    # Asserts type & value of the arguments.
    assert isinstance(language, str), "Variable language should be of type 'str'."
    assert language in [
        "es",
        "fr",
        "de",
    ], "Variable language should have value as 'es', 'fr', or 'de'."

    # A dictionary for the language-based Europarl dataset links.
    dataset_links = {
        "fr": "https://www.statmt.org/europarl/v7/fr-en.tgz",
        "de": "https://www.statmt.org/europarl/v7/de-en.tgz",
        "es": "https://www.statmt.org/europarl/v7/es-en.tgz",
    }

    # Download the compressed file.
    start_time = time.time()

    # Checks if the following directory path exists.
    dataset_directory_path = check_directory_path_existence("data/raw_data/europarl")

    # Checks if the file already exists. If yes, then does not download the file.
    file_path = "{}/{}-en.tgz".format(dataset_directory_path, language)
    if os.path.exists(file_path):
        print("{}-en.tgz already exists.".format(language))

    # Else, downloads it and saves it.
    else:
        # Sends request for the current language dataset file.
        response = requests.get(dataset_links[language])

        # Checks if the response has a success code. If not then prints the error message.
        assert response.status_code == 200, response.text

        # Saves the compressed file in the response as a .tgz file.
        with open(
            "{}/{}-en.tgz".format(dataset_directory_path, language), "wb"
        ) as out_file:
            out_file.write(response.content)
        out_file.close()

        print(
            "Finished downloading Europarl dataset for {}-en in {} sec.".format(
                language, round(time.time() - start_time, 3)
            )
        )
    print()


def download_paracrawl_dataset(language: str) -> None:
    """Downloads the Paracrawl dataset for the language given as input by user.

    Downloads the Paracrawl dataset for the language given as input by user.

    Args:
        language: A string for the language the Europarl dataset should be downloaded.

    Returns:
        None.
    """
    # Asserts type & value of the arguments.
    assert isinstance(language, str), "Variable language should be of type 'str'."
    assert language in [
        "es",
        "fr",
        "de",
    ], "Variable language should have value as 'es', 'fr', or 'de'."

    # Downloads paracrawl dataset into the corresponding directory for the current european language.
    start_time = time.time()
    _, _ = tfds.load(
        "para_crawl/en{}".format(language),
        split="train",
        with_info=True,
        shuffle_files=True,
        data_dir="data/raw_data/paracrawl/{}-en".format(language),
    )
    print(
        "Finished downloading paracrawl dataset for {}-en in {} sec.".format(
            language, round(time.time() - start_time, 3)
        )
    )
    print()
