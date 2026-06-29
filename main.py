from customer_segmentation.pipelines.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
    STAGE_NAME as STAGE_01_NAME,
)
from customer_segmentation.pipelines.stage_02_data_transformation import (
    DataTransformationPipeline,
    STAGE_NAME as STAGE_02_NAME,
)

# Stage 01: Data Ingestion
try:
    print(f">>>>>> Stage: {STAGE_01_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.initiate_data_ingestion()
    print(f">>>>>> Stage: {STAGE_01_NAME} completed <<<<<<\n\n{'=' * 50}")
except Exception as e:
    print(f"Stage failed: {e}")
    raise e

# Stage 02: Data Transformation
try:
    print(f">>>>>> Stage: {STAGE_02_NAME} started <<<<<<")
    obj = DataTransformationPipeline()
    obj.initiate_data_transformation()
    print(f">>>>>> Stage: {STAGE_02_NAME} completed <<<<<<\n\n{'=' * 50}")
except Exception as e:
    print(f"Stage failed: {e}")
    raise e
