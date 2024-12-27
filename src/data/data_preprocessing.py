import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
import os

# logging configuration
logger = logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

# File Handler
file_handler = logging.FileHandler("logs/data_preprocessing.log")
file_handler.setLevel("ERROR")

# Formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Adding handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(path: str) -> pd.DataFrame:
    """
    Load the data from the given path

    Args:
    path: The path to the csv file

    Returns:
    df: The dataframe
    """
    try:
        df = pd.read_csv(path)
        logger.debug("Loaded the dataframe from the given path")
        return df
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except Exception as e:
        logger.error(f"Error loading the dataframe: {e}")


# Define the preprocessing function
def preprocess_comment(comment: str):
    """
    Preprocess the comment

    Args:
    comment: The comment to preprocess

    Returns:
    comment: The preprocessed comment
    """
    # Convert to lowercase
    comment = comment.lower()

    # Remove trailing and leading whitespaces
    comment = comment.strip()

    # Remove newline characters
    comment = re.sub(r'\n', ' ', comment)

    # Remove non-alphanumeric characters, except punctuation
    comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

    # Remove stopwords but retain important ones for sentiment analysis
    stop_words = set(stopwords.words('english')) - \
        {'not', 'but', 'however', 'no', 'yet'}
    comment = ' '.join(
        [word for word in comment.split() if word not in stop_words])

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    comment = ' '.join([lemmatizer.lemmatize(word)
                       for word in comment.split()])

    return comment


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data

    Args:
    df: The dataframe

    Returns:
    df: The preprocessed dataframe
    """
    try:
        # Drop rows with missing values
        df.dropna(inplace=True)
        # Preprocess the comments
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
        df = df[~(df['clean_comment'].str.strip() == '')]
        logger.debug("Preprocessed the data")
        return df
    except Exception as e:
        logger.error(f"Error preprocessing the data: {e}")


def save_data(df: pd.DataFrame) -> None:
    """
    Save the dataframe to the given path

    Args:
    df: The dataframe
    """
    try:
        # Create path to store preprocessed data
        preprocessed_data_path = os.path.join("data", "processed")
        if not os.path.exists(preprocessed_data_path):
            os.makedirs(preprocessed_data_path)
        path = os.path.join(preprocessed_data_path, "sentiments_processed.csv")
        # Save the dataframe
        df.to_csv(path, index=False)
        logger.debug("Saved the dataframe to the given path")
    except Exception as e:
        logger.error(f"Error saving the dataframe: {e}")


if __name__ == "__main__":
    try:
        # Load the data
        df = load_data("data/raw/sentiments.csv")

        # Preprocess the data
        df = preprocess_data(df)

        # Save the data
        save_data(df)
    except Exception as e:
        logger.error(f"Error processing the data: {e}")
