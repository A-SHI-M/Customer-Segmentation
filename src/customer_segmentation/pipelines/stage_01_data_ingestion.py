from customer_segmentation.config.configuration import ConfigurationManager
from customer_segmentation.components.data_ingestion import DataIngestion

STAGE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_ingestion(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == "__main__":
    try:
        print(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.initiate_data_ingestion()
        print(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\n{'=' * 50}")
    except Exception as e:
        print(f"Stage failed: {e}")
        raise e
