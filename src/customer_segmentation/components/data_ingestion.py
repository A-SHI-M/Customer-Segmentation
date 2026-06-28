import os
import sys
import zipfile
import urllib.request
from customer_segmentation import logger
from customer_segmentation.utils import CustomException
from customer_segmentation.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> None:
        try:
            if not self.config.local_data_file.exists():
                logger.info(f"Downloading from: {self.config.source_url}")
                filename, _ = urllib.request.urlretrieve(
                    url=self.config.source_url,
                    filename=self.config.local_data_file,
                )
                logger.info(f"Download complete: {filename}")
            else:
                size = os.path.getsize(self.config.local_data_file)
                logger.info(f"File already exists ({size} bytes), skipping download.")
        except Exception as e:
            raise CustomException(e, sys) from e

    def extract_zip_file(self) -> None:
        try:
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)
            logger.info(f"Extracting zip to: {unzip_path}")
            with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
                zip_ref.extractall(unzip_path)
            os.remove(self.config.local_data_file)
            logger.info("Extraction complete. Zip file deleted.")
        except Exception as e:
            raise CustomException(e, sys) from e
