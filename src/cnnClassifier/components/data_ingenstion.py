import os
import tempfile
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier import logger


load_dotenv()


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def _make_s3(
        self,
        region_name: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
    ):
        return boto3.client(
            "s3",
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

    def _stream_download(
        self,
        s3,
        bucket_name: str,
        object_key: str,
        download_path: Path,
    ):
        download_path = Path(download_path)
        download_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Downloading {object_key} from bucket "
            f"{bucket_name} to {download_path}"
        )

        response = s3.get_object(
            Bucket=bucket_name,
            Key=object_key,
            RequestPayer="requester",
        )

        body = response["Body"]
        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                dir=download_path.parent,
                delete=False,
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)

                for chunk in body.iter_chunks(
                    chunk_size=8 * 1024 * 1024
                ):
                    if chunk:
                        tmp_file.write(chunk)

            os.replace(tmp_path, download_path)

            logger.info("Download completed successfully.")

        finally:
            body.close()

            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def download_from_s3(
        self,
        bucket_name: str,
        object_key: str,
        download_path: Path,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        region_name: str = "us-east-1",
    ):
        s3 = self._make_s3(
            region_name,
            aws_access_key_id,
            aws_secret_access_key,
        )

        try:
            self._stream_download(
                s3,
                bucket_name,
                object_key,
                download_path,
            )

        except ClientError as e:
            code = e.response["Error"].get("Code", "")
            msg = e.response["Error"].get(
                "Message",
                str(e),
            )

            logger.error(f"Error ({code}): {msg}")

            if code in ["403", "AccessDenied"]:
                logger.error(
                    "Check IAM permissions and ensure "
                    "RequestPayer='requester' is allowed."
                )
                raise

            if code in [
                "PermanentRedirect",
                "301",
                "AuthorizationHeaderMalformed",
            ]:
                try:
                    location = s3.get_bucket_location(
                        Bucket=bucket_name
                    )["LocationConstraint"]

                    retry_region = location or "us-east-1"

                    logger.info(
                        f"Detected bucket region: {retry_region}. "
                        "Retrying download..."
                    )

                    s3_retry = self._make_s3(
                        retry_region,
                        aws_access_key_id,
                        aws_secret_access_key,
                    )

                    self._stream_download(
                        s3_retry,
                        bucket_name,
                        object_key,
                        download_path,
                    )

                except ClientError as e2:
                    code2 = e2.response["Error"].get(
                        "Code",
                        "",
                    )
                    msg2 = e2.response["Error"].get(
                        "Message",
                        str(e2),
                    )

                    logger.error(
                        f"Retry failed ({code2}): {msg2}"
                    )
                    raise
            else:
                raise

    def extract_zip_file(self):
        """Extract the downloaded ZIP file."""
        unzip_path = Path(self.config.unzip_dir)
        unzip_path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(
            self.config.local_data_file,
            "r",
        ) as zip_ref:
            zip_ref.extractall(unzip_path)

        logger.info(
            f"Dataset extracted successfully to {unzip_path}"
        )