import logging
import os
from dotenv import load_dotenv
from dagshub import get_repo_bucket_client

# Load environment variables
load_dotenv()

# logging configuration
logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

# File Handler
file_handler = logging.FileHandler("logs/data_ingestion.log")
file_handler.setLevel("ERROR")

# Formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Adding handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Data Ingestion
def get_env():
    """
    Get the required environment variables
    """
    try:
        username = os.getenv("DAGSHUB_USERNAME")
        bucket = os.getenv("DAGSHUB_BUCKET")
        logger.debug("Retrieved environment variables")
        return username, bucket
    except Exception as e:
        logger.error(f"Error getting environment variables: {e}")


def connect_s3_bucket(username, bucket):
    """
    Connect to the S3 bucket

    Args:
    username: The username of the user
    bucket: The name of the bucket

    Returns:
    bucket_client: The S3 bucket client
    """
    try:
        bucket_path = f"{username}/{bucket}"
        bucket_client = get_repo_bucket_client(bucket_path)
        logger.debug("Connected to S3 bucket")
        return bucket_client
    except Exception as e:
        logger.error(f"Error connecting to S3 bucket: {e}")


def dowload_data(s3, bucket: str) -> None:
    """
    Download data from S3 bucket

    Args:
    s3: The S3 bucket client
    bucket: The name of the bucket
    """
    try:
        # Create path to store raw data
        raw_data_path = os.path.join("data", "raw")
        if not os.path.exists(raw_data_path):
            os.makedirs(raw_data_path)
        # Download data from S3 bucket
        s3.download_file(
            Bucket=bucket,
            Key="sentiments.csv",
            Filename="data/raw/sentiments.csv"
        )
        logger.debug("Downloaded data from S3 bucket")
    except Exception as e:
        logger.error(f"Error downloading data from S3 bucket: {e}")


if __name__ == "__main__":
    try:
        # Load environment variables
        env_vars = get_env()
        username = env_vars[0]
        bucket = env_vars[1]

        # Get s3 bucket client
        s3 = connect_s3_bucket(username, bucket)

        # Download data
        dowload_data(s3, bucket)

        # Log success
        logger.debug("Data ingestion step completed successfully")
    except Exception as e:
        logger.error(f"Error in data ingestion step: {e}")
