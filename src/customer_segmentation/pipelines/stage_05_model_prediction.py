from customer_segmentation.config.configuration import ConfigurationManager
from customer_segmentation.components.model_prediction import ModelPrediction

STAGE_NAME = "Model Prediction Stage"


class ModelPredictionPipeline:
    def __init__(self):
        pass

    def initiate_prediction(self):
        config = ConfigurationManager()
        pred_config = config.get_model_prediction_config()
        predictor = ModelPrediction(config=pred_config)
        return predictor.run()


if __name__ == "__main__":
    try:
        print(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        obj = ModelPredictionPipeline()
        obj.initiate_prediction()
        print(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\n{'=' * 50}")
    except Exception as e:
        print(f"Stage failed: {e}")
        raise e
