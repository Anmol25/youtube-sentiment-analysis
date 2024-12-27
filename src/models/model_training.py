import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
import scipy.sparse as sparse
import logging
import yaml
import pickle
import os

# logging configuration
logger = logging.getLogger("model_training")
logger.setLevel("DEBUG")

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

# File Handler
file_handler = logging.FileHandler("logs/model_training.log")
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
    except FileNotFoundError:
        logger.error(f"Error: The file '{path}' was not found.")
    except yaml.scanner.ScannerError as e:
        logger.error(f"Error: Invalid YAML syntax in '{path}'.\nDetails: {e}")
    except yaml.YAMLError as e:
        logger.error(f"Error: A YAML error occurred while parsing '{
                     path}'.\nDetails: {e}")
    except Exception as e:
        logger.error(
            f"Error: An unexpected error occurred while loading YAML file {e}")


def load_data(path: str, is_npz=False, is_npy=False):
    """
    Load the data from the given path

    Args:
    path: The path to the data file

    Returns:
    df: Data File
    """
    try:
        if is_npz:
            df = sparse.load_npz(path)
            logger.debug("Loaded the sparse data from the given path")
            return df
        elif is_npy:
            df = np.load(path)
            logger.debug("Loaded the numpy data from the given path")
            return df
        else:
            df = pd.read_csv(path)
            logger.debug("Loaded the data from the given path")
            return df
    except FileNotFoundError:
        logger.error(f"Error: The file '{path}' was not found.")
    except pd.errors.ParserError as e:
        logger.error(f"Error: An error occurred while parsing '{
                     path}'.\nDetails: {e}")
    except Exception as e:
        logger.error(
            f"Error: An unexpected error occurred while loading CSV file {e}")


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame,
                lor_params: dict, lgbm_params: dict,
                meta_params: dict) -> StackingClassifier:
    """
    Train a Stack Ensemble model using Logistic Regression and LightGBM

    Args:
    X_train: The training data
    y_train: The target data
    lor_params: The Logistic Regression hyperparameters
    lgbm_params: The LightGBM hyperparameters
    meta_params: The Meta model hyperparameters
    """
    try:
        # Initialize the base models
        base_models = [
            ("Logistic Regression", LogisticRegression(**lor_params)),
            ("LightGBM", LGBMClassifier(**lgbm_params))
        ]

        # Initialize the meta model
        meta_model = LogisticRegression(**meta_params)

        # Initialize the stack ensemble model
        stack_model = StackingClassifier(estimators=base_models,
                                         final_estimator=meta_model,
                                         cv=5, n_jobs=-1)

        # Train the model
        stack_model.fit(X_train, y_train)
        logger.info("Model training completed successfully")
        return stack_model
    except Exception as e:
        logger.error(f"Error in Training Model: {e}")


def save_model(model: StackingClassifier) -> None:
    """
    Save the model to the given path

    Args:
    model: The model to save
    """
    try:
        model_path = os.path.join("models", "model")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        path = os.path.join(model_path, "model.pkl")
        with open(path, "wb") as file:
            pickle.dump(model, file)
        logger.info("Model saved successfully")
    except Exception as e:
        logger.error(f"Error in Saving Model: {e}")


if __name__ == "__main__":
    try:
        # Load the parameters
        params = load_params("params.yaml")

        # Load model Hyperparameters
        lor_params = params["model_training"]["LoR"]
        lgbm_params = params["model_training"]["LightGBM"]
        meta_params = params["model_training"]["Meta"]

        # Train Data
        X_train = load_data("data/interim/X_train.npz", is_npz=True)
        y_train = load_data("data/interim/y_train.npy", is_npy=True)

        # Train model
        model = train_model(X_train, y_train, lor_params, lgbm_params,
                            meta_params)

        # Save the model
        save_model(model)

    except Exception as e:
        print(f"Error: {e}")
