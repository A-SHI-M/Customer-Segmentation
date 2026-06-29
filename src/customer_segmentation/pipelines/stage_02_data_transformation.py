from customer_segmentation.config.configuration import ConfigurationManager
from customer_segmentation.components.data_transformation import DataTransformation

STAGE_NAME = "Data Transformation Stage"


class DataTransformationPipeline:
    def __init__(self):
        pass

    def initiate_data_transformation(self):
        config = ConfigurationManager()
        dt_config = config.get_data_transformation_config()
        dt = DataTransformation(config=dt_config)
        dt.run()


if __name__ == "__main__":
    try:
        print(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        obj = DataTransformationPipeline()
        obj.initiate_data_transformation()
        print(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\n{'=' * 50}")
    except Exception as e:
        print(f"Stage failed: {e}")
        raise e
