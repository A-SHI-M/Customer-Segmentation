from customer_segmentation.pipelines.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
    STAGE_NAME,
)

try:
    print(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.initiate_data_ingestion()
    print(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\n{'=' * 50}")
except Exception as e:
    print(f"Stage failed: {e}")
    raise e
