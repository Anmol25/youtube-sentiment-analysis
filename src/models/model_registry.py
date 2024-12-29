import json
import mlflow
import dagshub
from dotenv import load_dotenv
import logging
import os

# Load env variables
load_dotenv()

# logging configuration
logger = logging.getLogger("model_registry")
logger.setLevel("DEBUG")

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

# File Handler
file_handler = logging.FileHandler("logs/model_registry.log")
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


def load_model_info(file_path: str) -> dict:
    """
    Load the model info from a JSON file.

    Args:
    file_path: Path to model info

    Returns:
    model_info: Info about model
    """
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error(
            'Unexpected error occurred while loading the model info: %s', e)
        raise


def register_model(model_name: str, model_info: dict):
    """
    Register the model to the MLflow Model Registry.

    Args:
    model_name: Name of model
    model_info: Model info like run_id and path
    """
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)

        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        client.set_model_version_tag(
            name=model_name,
            version=model_version.version,
            key="stage",
            value="Staging"
        )
        logger.debug(f"Model {model_name} version {model_version.version}"
                     "registered and transitioned to Staging.")
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise


if __name__ == "__main__":
    try:
        # Initialize mlflow
        mlflow_init()
        # Get Model Info
        model_info_path = 'models/info/experiment_info.json'
        model_info = load_model_info(model_info_path)
        # Register Model
        model_name = "yt_chrome_plugin_model"
        register_model(model_name, model_info)
    except Exception as e:
        logger.error(
            'Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")
