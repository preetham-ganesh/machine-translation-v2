import os
import sys
import warnings
import argparse
import logging


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.FATAL)


from bs4 import BeautifulSoup


def remove_html_markup(self, text: str) -> str:
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
