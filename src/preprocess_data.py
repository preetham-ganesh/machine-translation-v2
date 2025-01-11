import os
import sys
import warnings
import argparse
import logging
import unicodedata
import re
import zipfile


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.FATAL)


from bs4 import BeautifulSoup

from src.utils import check_directory_path_existence

from typing import Dict


# A dictionary for unique word count in each language.
unique_word_count = {"en": dict(), "es": dict(), "fr": dict(), "de": dict()}


def extract_data_from_zip_file(language: str) -> None:
    """Extracts files from downloaded Tatoeba dataset zip file.

    Extracts files from downloaded Tatoeba dataset zip file.

    Args:
        language: A string for the language the dataset belongs to.

    Returns:
        None.
    """
    # Asserts type & values of the arguments.
    assert isinstance(language, str), "Variable language should be of type 'str'."

    # Creates absolute directory path for downloaded data zip file.
    home_directory_path = os.getcwd()
    zip_file_path = os.path.join(
        home_directory_path, f"data/raw_data/tatoeba/{language}-en.zip"
    )

    # Creates the directory path.
    extracted_data_directory_path = check_directory_path_existence(
        "data/extracted_data/tatoeba"
    )

    # A dictionary for the name of the text files in each language.
    text_name = {"fr": "fra.txt", "de": "deu.txt", "es": "spa.txt"}

    # If file does not exist, then extracts files from the directory.
    if not os.path.exists(
        os.path.join(
            extracted_data_directory_path, f"/{language}-en/{text_name[language]}.csv"
        )
    ):

        # Extracts files from downloaded data zip file into a directory.
        try:
            with zipfile.ZipFile(zip_file_path, "r") as zip_file:
                zip_file.extractall(extracted_data_directory_path)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"{zip_file_path} does not exist. Run 'download_data.py' to download the data."
            )
        print(
            f"Finished extracting files from 'archive.zip' to {extracted_data_directory_path}."
        )
        print()


def remove_html_markup(text: str) -> str:
    """Removes HTML markup components from text.

    Removes HTML markup components from text.

    Args:
        text: A string for the text which needs to be processed.

    Returns:
        A string for the processed text without HTML markup components.
    """
    # Asserts type & values of the arguments.
    assert isinstance(text, str), "Variable text should be of type 'str'."

    # Creates an object for BeautifulSoup.
    soup = BeautifulSoup(text, "lxml")

    # Get the text content of all visible elements.
    text = soup.get_text(strip=True)

    # Remove any leading or trailing spaces.
    text = text.strip()

    # Replace consecutive whitespace characters with a single space.
    return " ".join(text.split())


def preprocess_text(
    text: str, language: str, n_max_words_per_text: int, update_word_count: bool
) -> str:
    """Preprocesses text to remove unwanted characters from it.

    Preprocesses text to remove unwanted characters from it.

    Args:
        text: A string for the text which needs to be processed.
        language: A string for the name of the language the dataset belongs to.
        n_max_words_per_text: An integer for the maximum number of words allowed in a text.
        update_unique_words: A boolean value for updating the word count.

    Returns:
        A string for processed version of input text.
    """
    # Asserts type & values of the arguments.
    assert isinstance(text, str), "Variable text should be of type 'str'."
    assert isinstance(language, str), "Variable language should be of type 'str'."
    assert language in [
        "en",
        "es",
        "fr",
        "de",
    ], "Variable language should have value as 'en', 'es', 'fr', or 'de'."
    assert isinstance(
        n_max_words_per_text, int
    ), "Variable n_max_words_per_text should be of type 'int'."
    assert isinstance(
        update_word_count, bool
    ), "Variable update_word_count should be of type 'bool'."

    # Removes HTML markup components from text provided as input.
    text = remove_html_markup(text)

    # Converts text to lowercase characters, & strip leading & trailing whitespace.
    text = text.lower().strip()

    # Replaces unwanted characters in text.
    text = text.replace("##at##-##at##", "-")
    text = text.replace("&apos;", "'")
    text = text.replace("&quot;", '"')
    text = text.replace("&#91;", "")
    text = text.replace("&#93;", "")
    text = text.replace("&#124;", "")
    text = text.replace('"', ' " ')

    # Based on name of the language, removes characters from text.
    if language == "en":
        text = "".join(
            id_0
            for id_0 in unicodedata.normalize("NFKD", str(text))
            if unicodedata.category(id_0) != "Mn"
        )
        text = re.sub(r"[^-!$&(),./%0-9:;?a-z€'\"]+", " ", text)

    elif language == "es":
        text = re.sub(r"[^-!$&(),./%0-9:;?ÁÉÍÓÚÑÜáéíóúñü¿¡a-z€'\"]+", " ", text)

    elif language == "fr":
        text = re.sub(r"[^-!$&(),./%0-9:;?!çàâæéèêëîïôöûüù'€\"*]+", " ", text)

    elif language == "de":
        text = re.sub(r"[^-!$&(),./%0-9:;?!äöüßœáéíóúñüa-z'€\"*]+", " ", text)

    # Collapses repeated punctuation.
    text = re.sub(r"\.{2,}", ".", text)

    # Fix ordinal numbers (e.g., 1st, 2nd)
    text = re.sub(r"(\d)th", r"\1 th", text, flags=re.I)
    text = re.sub(r"(\d)st", r"\1 st", text, flags=re.I)
    text = re.sub(r"(\d)rd", r"\1 rd", text, flags=re.I)
    text = re.sub(r"(\d)nd", r"\1 nd", text, flags=re.I)

    # Separate punctuation with spaces
    punctuations = "-!$&(),./%:;?!çàâæéèêëîïôöûüù'€\"*"
    for character in punctuations:
        text = text.replace(character, " " + character + " ")

    # Splits text into list of words as strings.
    text_words = text.split(" ")

    # If no. of words in current text is more than maximum limit, then text is ignored.
    if len(text_words) > n_max_words_per_text:
        return ""

    # Iterates across words in text.
    filtered_words = list()
    for word in text_words:
        # If word is not empty, then it is appended to list.
        if word != "":
            filtered_words.append(word)

            # Unique word count is updated for current word.
            if update_word_count:
                unique_word_count[language][word] = 1 + unique_word_count[language].get(
                    word, 0
                )

    # Converts of list of filtered words into a single string.
    filtered_text = " ".join(filtered_words)
    return filtered_text


def preprocess_tatoeba_dataset(language: str) -> None:
    """Preprocesses the Tatoeba dataset for the language given as input by user.

    Preprocesses the Tatoeba dataset for the language given as input by user.

    Args:
        language: A string for the language the Tatoeba dataset should be preprocessed.

    Returns:
        None.
    """
