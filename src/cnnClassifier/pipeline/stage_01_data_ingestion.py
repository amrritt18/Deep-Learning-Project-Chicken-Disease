import os

from dotenv import load_dotenv

from cnnClassifier import logger
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier.config.configuration import ConfigurationManager


load_dotenv()

STAGE_NAME = "Data Ingestion stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

            if not aws_access_key_id or not aws_secret_access_key:
                raise ValueError(
                    "AWS credentials are not configured. "
                    "Add AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
                    "to the .env file before running data ingestion."
                )

            config = ConfigurationManager()

            data_ingestion_config = (
                config.get_data_ingestion_config()
            )

            data_ingestion = DataIngestion(
                config=data_ingestion_config
            )

            data_ingestion.download_from_s3(
                bucket_name=data_ingestion_config.bucket_name,
                object_key="data.zip",
                download_path=data_ingestion_config.local_data_file,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name="us-east-1",
            )

            data_ingestion.extract_zip_file()

            logger.info(
                f"{STAGE_NAME} completed successfully."
            )

        except Exception as e:
            logger.exception(
                f"{STAGE_NAME} failed."
            )
            raise e