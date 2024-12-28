import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import mlflow
from mlflow.models.signature import infer_signature
from dotenv import load_dotenv
import logging
import pickle
import yaml
import json
import os

# Load env variables
load_dotenv()

# logging configuration
logger = logging.getLogger("model_evaluation")
logger.setLevel("DEBUG")

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

# File Handler
file_handler = logging.FileHandler("logs/model_evaluation.log")
file_handler.setLevel("ERROR")

# Formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Adding handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def mlflow_init():
    """
    Initialize MLFlow Tracking Server
    """
    try:
        # Initialize Dagshub
        DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
        DAGSHUB_BUCKET = os.getenv("DAGSHUB_BUCKET")
        dagshub.init(repo_owner=DAGSHUB_USERNAME,
                     repo_name=DAGSHUB_BUCKET, mlflow=True)
        # Initialize mlflow tracking server
        mlflow.set_tracking_uri(
            f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_BUCKET}.mlflow")
        logger.debug("Succesfully initialized mlfow tracking server")
    except Exception as e:
        logger.error(
            f"Error occured in initializing mlflow tracking server: {e}")


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


def load_model(path: str):
    """
    Load the model from the given path

    Args:
    path: The path to the model file

    Returns:
    model: The model
    """
    try:
        with open(path, "rb") as file:
            model = pickle.load(file)
            logger.debug("Loaded the model from the given path")
            return model
    except FileNotFoundError:
        logger.error(f"Error: The file '{path}' was not found.")
    except pickle.UnpicklingError as e:
        logger.error(f"Error: An error occurred while unpickling '{
                     path}'.\nDetails: {e}")
    except Exception as e:
        logger.error(
            f"Error: An unexpected error occurred while loading the model {e}")


def log_params(params: dict):
    """
    Log Parameters

    Args:
    params: dictionary of parameters
    """
    try:
        # Log Params
        for stage, stage_params in params.items():
            for type, type_params in params[stage].items():
                for key, value in params[stage][type].items():
                    mlflow.log_param(f"{type}_{key}", value)
        logger.debug("Parameters Logged Successfully")
    except Exception as e:
        logger.error(
            f"Error: Unexpected error occured while logging params: {e}")


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test data

    Args:
    model: The model
    X_test: The test features
    y_test: The test target

    Returns:
    report: The classification report
    cm : Confusion Matrix
    accuracy: Accuracy of model
    """
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        logger.debug("Model evaluated successfully")
        return report, cm, accuracy
    except Exception as e:
        logger.error(
            f"Error: Unexpected error occured while loadinf model {e}")


def save_model_info(run_id: str, model_path: str) -> None:
    """
    Save the model run ID and path to a JSON file.
    """
    try:
        # Create a dictionary with the info you want to save
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }
        model_path = os.path.join("models", "info")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        path = os.path.join(model_path, "experiment_info.json")
        # Save the dictionary as a JSON file
        with open(path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise


def log_report(report):
    """
    Save the classification report to the given path and log metrics

    Args:
    report: The classification report
    """
    try:
        # Make the path if it does not exist
        if not os.path.exists("reports"):
            os.makedirs("reports")
        path = os.path.join("reports", "metrics.json")
        # Save the report as a json file
        with open(path, "w") as file:
            json.dump(report, file)

        # Log classification report metrics for the test data
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                mlflow.log_metrics({
                    f"test_{label}_precision": metrics['precision'],
                    f"test_{label}_recall": metrics['recall'],
                    f"test_{label}_f1-score": metrics['f1-score']
                })
        logger.debug("Report saved successfully")
    except FileNotFoundError:
        logger.error(f"Error: The file '{path}' was not found.")
    except Exception as e:
        logger.error(
            f"Error: An unexpected error occurred while saving the report {e}")


def log_cm(cm: dict):
    """
    Make a confusion metric figure and log artifact

    Args:
    cm : Confustion matrix
    """
    try:
        # Create Confusion matrix figure
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix: Stacking Classifier")
        plt.savefig("reports/figures/confusion_matrix.png")
        mlflow.log_artifact("reports/figures/confusion_matrix.png")
        plt.close()
        logger.debug("Successfully logged Confusion matrix")
    except Exception as e:
        logger.error(f"Error in logging confusion matrix: {e}")


if __name__ == "__main__":
    try:
        # Start mlflow tracking server
        mlflow_init()

        # MLFlow Experiment tracking
        mlflow.set_experiment("Model Evaluation in DVC Pipeline")

        with mlflow.start_run() as run:
            mlflow.set_tag("mlflow.runName",
                           "Stacked Classifier of LoR and LightGBM")
            mlflow.set_tag("Stacking", "LoR +LightGBM")

            params = load_params("params.yaml")
            # Log Params
            log_params(params)

            # Load the model
            model = load_model("models/model/model.pkl")

            # Load the test data
            X_test = load_data("data/interim/X_test.npz", is_npz=True)
            y_test = load_data("data/interim/y_test.npy", is_npy=True)

            # Evaluate model
            report, cm, accuracy = evaluate_model(model, X_test, y_test)

            # Log classification report metrics for the test data
            log_report(report)
            log_cm(cm)
            mlflow.log_metric("accuracy", accuracy)

            # Log vectorizer
            mlflow.log_artifact("models/vectorizer/vectorizer.pkl")

            # Log Model
            input_example = X_test[:5]
            signature = infer_signature(input_example, y_test[:5])
            # Log model
            model_path = "Stacked_Model"
            mlflow.sklearn.log_model(
                model,
                model_path,
                input_example=input_example,
                signature=signature)
            save_model_info(run.info.run_id, model_path)
            mlflow.end_run()

        logger.debug("Model evaluation completed successfully")
    except Exception as e:
        logger.error(f"Error: Unexpected error in model evaluation {e}")
