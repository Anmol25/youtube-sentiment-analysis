import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import scipy.sparse as sparse
import logging
import yaml
import pickle
import os

# logging configuration
logger = logging.getLogger("feature_engineering")
logger.setLevel("DEBUG")

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

# File Handler
file_handler = logging.FileHandler("logs/feature_engineering.log")
file_handler.setLevel("ERROR")

# Formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Adding handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(path: str) -> dict:
    """
    Load the parameters from the given path

    Args:
    path: The path to the yaml file

    Returns:
    params: The parameters
    """
    try:
        with open(path, "r") as file:
            params = yaml.safe_load(file)
        logger.debug("Loaded the parameters from the given path")
        return params
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except Exception as e:
        logger.error(f"Error loading the parameters: {e}")


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


def vectorize(df: pd.DataFrame, max_features: int,
              ngram_range: tuple, test_size: float,
              random_state: int) -> tuple:
    """
    Vectorize the comments using TF-IDF and
    split the data into train and test sets

    Args:
    df: The dataframe
    max_features: The maximum number of features
    ngram_range: The ngram range
    test_size: Test size for train test split
    random_state: Random state for train test spit

    Returns:
    X_train, X_test, y_train, y_test: The train and test splits
    """
    try:
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            df['clean_comment'], df['category'], test_size=test_size,
            random_state=random_state, stratify=df['category'])

        # Vectorize the comments using TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range)
        # Apply Vectorizer
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)
        logger.debug("Vectorized the comments using TF-IDF")
        return X_train, X_test, y_train, y_test, vectorizer
    except Exception as e:
        logger.error(f"Error vectorizing the comments: {e}")


def save_vectorizer(vectorizer: TfidfVectorizer) -> None:
    """
    Save the vectorizer to the given path

    Args:
    vectorizer: The vectorizer
    """
    try:
        # Create Path to store the vectorizer
        vectorizer_path = os.path.join("models", "vectorizer")
        if not os.path.exists(vectorizer_path):
            os.makedirs(vectorizer_path)
        # Make Path to Store the Vectorizer
        path = os.path.join(vectorizer_path, "vectorizer.pkl")
        with open(path, "wb") as file:
            pickle.dump(vectorizer, file)
        logger.debug("Saved the vectorizer to the given path")
    except Exception as e:
        logger.error(f"Error saving the vectorizer: {e}")


def save_data(X_train, X_test, y_train, y_test) -> None:
    """
    Save the sparse matrices and target variables in efficient formats

    Args:
    X_train: Train Data (sparse matrix)
    X_test: Test Data (sparse matrix)
    y_train: Train Target
    y_test: Test Target
    """
    try:
        # Create Path to store the data
        data_path = os.path.join("data", "interim")
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        # Save sparse matrices in npz format
        sparse.save_npz(os.path.join(data_path, "X_train.npz"), X_train)
        sparse.save_npz(os.path.join(data_path, "X_test.npz"), X_test)

        # Save target variables as numpy arrays
        np.save(os.path.join(data_path, "y_train.npy"), y_train)
        np.save(os.path.join(data_path, "y_test.npy"), y_test)

        logger.debug("Saved the data in sparse format")
    except Exception as e:
        logger.error(f"Error saving the data: {e}")


if __name__ == "__main__":
    try:
        # Load the parameters
        params = load_params("params.yaml")
        max_features = params["feature_engineering"]["vectorizer"]["max_features"]
        ngram_range = tuple(params["feature_engineering"]
                            ["vectorizer"]["ngram_range"])
        test_size = params["feature_engineering"]["train_test_split"]["test_size"]
        random_state = params["feature_engineering"]["train_test_split"]["random_state"]

        # Load the data
        df = load_data("data/processed/sentiments_processed.csv")
        # Split data and Vectorize
        X_train, X_test, y_train, y_test, vectorizer = vectorize(
            df, max_features, ngram_range, test_size, random_state)

        # Save the Vectorizer
        save_vectorizer(vectorizer)
        # Save the Data
        save_data(X_train, X_test, y_train, y_test)
        logger.debug("Feature Engineering completed")
    except Exception as e:
        logger.error(f"Error loading the parameters: {e}")
