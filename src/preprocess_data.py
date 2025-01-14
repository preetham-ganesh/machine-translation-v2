import os
import sys
import warnings
import argparse
import logging
import unicodedata
import re
import zipfile
import tarfile


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.FATAL)


from bs4 import BeautifulSoup
import pandas as pd
import tensorflow_datasets as tfds

from src.utils import check_directory_path_existence, save_text_file, load_text_file
from src.download_data import check_language

from typing import List


class PreprocessText(object):
    """"""

    def __init__(self, language: str, n_max_words_per_text: int) -> None:
        """Creates object attributes for the PreprocessText class.

        Creates object attributes for the PreprocessText class.

        Args:
            language: A string for the name of the european language the text belongs to.
            n_max_words_per_text: An integer for the maximum no. of the words in a text.

        Returns:
            None.
        """
        # Asserts type & values of the arguments.
        check_language(language)
        assert isinstance(
            n_max_words_per_text, int
        ), "Variable n_max_words_per_text should be of type 'int'."

        # Initializes class variables.
        self.language = language
        self.n_max_words_per_text = 50
        self.unique_word_count = {"en": dict(), language: dict()}
        self.rare_words = {"en": set(), self.language: set()}

    def extract_tatoeba_dataset(self) -> None:
        """Extracts the Tatoeba dataset for the language given as input by user.

        Extracts the Tatoeba dataset for the language given as input by user.

        Args:
            None.

        Returns:
            None.
        """
        # Creates absolute directory path for downloaded data zip file.
        zip_file_path = os.path.join(
            BASE_PATH, f"data/raw_data/tatoeba/{self.language}-en.zip"
        )

        # Creates the directory path.
        extracted_data_directory_path = check_directory_path_existence(
            f"data/extracted_data/tatoeba/{self.language}-en"
        )

        # A dictionary for the name of the text files in each language.
        text_name = {"fr": "fra.txt", "de": "deu.txt", "es": "spa.txt"}

        # If file does not exist, then extracts files from the directory.
        if not os.path.exists(
            os.path.join(extracted_data_directory_path, text_name[self.language])
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
                f"Finished extracting files from '{self.language}-en.zip' to {extracted_data_directory_path}."
            )
            print()

    def extract_europarl_dataset(self) -> None:
        """Extracts the Europarl dataset for the language given as input by user.

        Extracts the Europarl dataset for the language given as input by user.

        Args:
            None.

        Returns:
            None.
        """
        # Creates absolute directory path for downloaded data tar file.
        tar_file_path = os.path.join(
            BASE_PATH, f"data/raw_data/europarl/{self.language}-en.tgz"
        )

        # Creates the directory path.
        extracted_data_directory_path = check_directory_path_existence(
            f"data/extracted_data/europarl/{self.language}-en"
        )

        # If file does not exist, then extracts files from the directory.
        if not os.path.exists(
            os.path.join(
                extracted_data_directory_path,
                f"europarl-v7.{self.language}-en.{self.language}",
            )
        ):
            # Extracts files from downloaded data tar file into a directory.
            try:
                file = tarfile.open(tar_file_path)
                file.extractall(extracted_data_directory_path)
                file.close()
            except FileNotFoundError as error:
                raise FileNotFoundError(
                    f"{tar_file_path} does not exist. Run 'download_data.py' to download the data."
                )
            print(
                f"Finished extracting files from '{self.language}-en.tgz' to {extracted_data_directory_path}."
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
                unique_words_count[language][word] = 1 + unique_words_count[
                    language
                ].get(word, 0)

    # Converts of list of filtered words into a single string.
    filtered_text = " ".join(filtered_words)
    return filtered_text


def preprocess_tatoeba_dataset(
    language: str,
    n_max_words_per_text: int,
) -> None:
    """Preprocesses the Tatoeba dataset for the language given as input by user.

    Preprocesses the Tatoeba dataset for the language given as input by user.

    Args:
        language: A string for the language the Tatoeba dataset should be preprocessed.
        n_max_words_per_text: An integer for the maximum number of words allowed in a text.

    Returns:
        None.
    """
    # Checks if the language is valid or not.
    check_language(language)
    assert isinstance(
        n_max_words_per_text, int
    ), "Variable n_max_words_per_text should be of type 'int'."

    # A dictionary for the name of the text files in each language.
    text_name = {"fr": "fra.txt", "de": "deu.txt", "es": "spa.txt"}

    # Loads the Tatoeba dataset for the language given as input by user.
    data = pd.read_csv(
        os.path.join(
            BASE_PATH,
            f"data/extracted_data/tatoeba/{language}-en",
            text_name[language],
        ),
        sep="\t",
        encoding="utf-8",
        names=["en", language, "x"],
    )
    print(f"No. of original {language}-en pairs in Tatoeba dataset: {len(data)}")
    print()

    # Iterates across rows in the dataset.
    n_processed_pairs = 0
    for id_0, row in data.iterrows():

        # Preprocesses the text in the dataset.
        en_text = preprocess_text(str(row["en"]), "en", n_max_words_per_text)
        eu_text = preprocess_text(str(row[language]), language, n_max_words_per_text)

        # If text is not empty, then it is appended to list.
        if en_text != "" and eu_text != "":
            processed_texts.append(
                {"en": en_text, language: eu_text, "dataset": "tatoeba"}
            )
            n_processed_pairs += 1

        if id_0 % 1000 == 0:
            print(
                f"Finished processing {round((id_0 / len(data)) * 100, 3)}% {language}-en pairs in Tatoeba dataset."
            )
    print()
    print(
        f"No. of processed {language}-en pairs in Tatoeba dataset: {n_processed_pairs}"
    )
    print()


def preprocess_europarl_dataset(language: str, n_max_words_per_text: int) -> None:
    """Preprocesses the Europarl dataset for the language given as input by user.

    Preprocesses the Europarl dataset for the language given as input by user.

    Args:
        language: A string for the language the Europarl dataset should be preprocessed.
        n_max_words_per_text: An integer for the maximum number of words allowed in a text.

    Returns:
        None.
    """
    # Checks if the language is valid or not.
    check_language(language)
    assert isinstance(
        n_max_words_per_text, int
    ), "Variable n_max_words_per_text should be of type 'int'."

    # Loads the Europarl dataset for the language given as input by user.
    original_en_texts = load_text_file(
        f"europarl-v7.{language}-en.en",
        os.path.join(BASE_PATH, f"data/extracted_data/europarl/{language}-en"),
    ).split("\n")
    original_eu_texts = load_text_file(
        f"europarl-v7.{language}-en.{language}",
        os.path.join(BASE_PATH, f"data/extracted_data/europarl/{language}-en"),
    ).split("\n")

    # Checks if the length of both the lists are equal.
    assert len(original_en_texts) == len(
        original_eu_texts
    ), f"Length of en and {language} texts should be equal."
    print(
        f"No. of original {language}-en pairs in Europarl dataset: {len(original_en_texts)}"
    )
    print()

    # Iterates across rows in the dataset.
    n_processed_pairs = 0
    for id_0 in range(len(original_en_texts)):
        # Preprocesses the text in the dataset.
        en_text = preprocess_text(original_en_texts[id_0], "en", n_max_words_per_text)
        eu_text = preprocess_text(
            original_eu_texts[id_0], language, n_max_words_per_text
        )

        # If text is not empty, then it is appended to list.
        if en_text != "" and eu_text != "":
            processed_texts.append(
                {"en": en_text, language: eu_text, "dataset": "europarl"}
            )
            n_processed_pairs += 1

        if id_0 % 1000 == 0:
            print(
                f"Finished processing {round((id_0 / len(original_en_texts)) * 100, 3)}% {language}-en pairs in "
                + "Europarl dataset."
            )
    print()
    print(
        f"No. of processed {language}-en pairs in Europarl dataset: {n_processed_pairs}"
    )
    print()


def preprocess_paracrawl_dataset(language: str, n_max_words_per_text: int) -> None:
    """Preprocesses the Paracrawl dataset for the language given as input by user.

    Preprocesses the Paracrawl dataset for the language given as input by user.

    Args:
        language: A string for the language the Paracrawl dataset should be preprocessed.
        n_max_words_per_text: An integer for the maximum number of words allowed in a text.

    Returns:
        None.
    """
    # Checks if the language is valid or not.
    check_language(language)
    assert isinstance(
        n_max_words_per_text, int
    ), "Variable n_max_words_per_text should be of type 'int'."

    # Loads the Paracrawl dataset for the language given as input by user.
    dataset, info = tfds.load(
        f"para_crawl/en{language}_plain_text".format(language),
        split="train",
        with_info=True,
        shuffle_files=True,
        data_dir=f"{BASE_PATH}/data/raw_data/paracrawl/{language}-en",
    )
    n_rows = info.splits["train"].num_examples
    print(f"No. of original {language}-en pairs in Paracrawl dataset: {n_rows}")

    # Iterates across rows in the dataset.
    n_processed_pairs = 0
    for id_0, row in enumerate(dataset):

        # Preprocesses the text in the dataset.
        en_text = preprocess_text(
            row["en"].numpy().decode("utf-8"), "en", n_max_words_per_text
        )
        eu_text = preprocess_text(
            row[language].numpy().decode("utf-8"), language, n_max_words_per_text
        )

        # If text is not empty, then it is appended to list.
        if en_text != "" and eu_text != "":
            processed_texts.append(
                {"en": en_text, language: eu_text, "dataset": "paracrawl"}
            )
            n_processed_pairs += 1

        if id_0 % 1000 == 0:
            print(
                f"Finished processing {round((id_0 / n_rows) * 100, 3)}% {language}-en pairs in "
                + "Paracrawl dataset."
            )
    print()
    print(
        f"No. of processed {language}-en pairs in Paracrawl dataset: {n_processed_pairs}"
    )
    print()


def identify_language_rare_words(language: str) -> None:
    """Identifies rare words for the language in the dataset.

    Identifies rare words for the language in the dataset.

    Args:
        language: A string for the language the rare words should be identified for.

    Returns:
        None.
    """
    # Checks if the language is valid or not.
    check_language(language)

    print(
        f"No. of unique words for {language} language in the dataset: {len(unique_words_count[language])}"
    )

    # Iterates across unique words in the dataset based on language.
    for word in unique_words_count[language].keys():
        # If count of word is 1, word consists of only alphabets, and length of word is between 8 & 10,
        # then word is added to rare words list.
        if (
            unique_words_count[language][word] == 1
            and word.isalpha()
            and not word.isdigit()
            and len(word) > 7
            and len(word) < 11
        ):
            rare_words[language].add(word)
    print(
        f"No. of rare words for {language} language in the dataset: {len(rare_words[language])}"
    )
    print()


def identify_common_rare_words(language: str) -> List[str]:
    """Identifies common rare words between en & the european language.

    Identifies common rare words between en & the european language.

    Args:
        language: A string for the language the common rare words should be identified for.

    Returns:
        A list of common rare words between en & the european language.
    """
    # Checks if the language is valid or not.
    check_language(language)

    # Identifies rare words for each language in the dataset.
    identify_language_rare_words("en")
    identify_language_rare_words(language)

    # Identifies common rare words between en & the european language.
    common_rare_words = set(rare_words[language] & rare_words["en"])
    print(
        f"No. of common rare words between en & {language} texts: {len(common_rare_words)}"
    )


def main():
    print()

    # Parses the arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        required=True,
        help="Enter name of language for which datasets should be downloaded. Current options: 'es', fr' or 'de'.",
    )
    parser.add_argument(
        "-nmwpt",
        "--n_max_words_per_text",
        type=int,
        required=True,
        help="Enter no. of maximum words allowed in a text.",
    )
    parser.add_argument(
        "-dv",
        "--dataset_version",
        type=str,
        required=True,
        help="Enter the version by which the processed dataset should be saved as.",
    )
    args = parser.parse_args()  # Add arg for dataset size.

    # Checks if the arguments, have valid values.
    assert args.language in [
        "es",
        "fr",
        "de",
    ], "Argument language should have value as 'es', 'fr', or 'de'."

    # Extracts the Tatoeba dataset for the language given as input by user.
    extract_tatoeba_dataset(args.language)

    # Extracts the Europarl dataset for the language given as input by user.
    extract_europarl_dataset(args.language)

    # Creates global variables for processed texts, unique words count & rare words.
    global processed_texts, unique_words_count, rare_words
    processed_texts = list()
    unique_words_count = {"en": dict(), args.language: dict(), "dataset": dict()}
    rare_words = {"en": set(), args.language: set()}

    # Preprocesses the Tatoeba dataset for the language given as input by user.
    preprocess_tatoeba_dataset(
        args.language, args.n_max_words_per_text, args.dataset_version
    )

    # Preprocesses the Europarl dataset for the language given as input by user.
    preprocess_europarl_dataset(
        args.language, args.n_max_words_per_text, args.dataset_version
    )

    # Preprocesses the Paracrawl dataset for the language given as input by user.
    preprocess_paracrawl_dataset(
        args.language, args.n_max_words_per_text, args.dataset_version
    )


if __name__ == "__main__":
    main()
