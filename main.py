import sys
sys.stdout.reconfigure(encoding="utf-8")

from customer_segmentation.pipelines.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
    STAGE_NAME as STAGE_01_NAME,
)
from customer_segmentation.pipelines.stage_02_data_transformation import (
    DataTransformationPipeline,
    STAGE_NAME as STAGE_02_NAME,
)
from customer_segmentation.pipelines.stage_03_model_trainer import (
    ModelTrainerPipeline,
    STAGE_NAME as STAGE_03_NAME,
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

# Stage 03: Model Training
try:
    print(f">>>>>> Stage: {STAGE_03_NAME} started <<<<<<")
    obj = ModelTrainerPipeline()
    obj.initiate_model_training()
    print(f">>>>>> Stage: {STAGE_03_NAME} completed <<<<<<\n\n{'=' * 50}")
except Exception as e:
    print(f"Stage failed: {e}")
    raise e
