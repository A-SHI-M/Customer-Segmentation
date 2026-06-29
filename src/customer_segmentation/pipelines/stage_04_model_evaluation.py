from customer_segmentation.config.configuration import ConfigurationManager
from customer_segmentation.components.model_evaluation import ModelEvaluation

STAGE_NAME = "Model Evaluation Stage"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def initiate_model_evaluation(self):
        config = ConfigurationManager()
        eval_config = config.get_model_evaluation_config()
        evaluator = ModelEvaluation(config=eval_config)
        evaluator.run()


if __name__ == "__main__":
    try:
        print(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.initiate_model_evaluation()
        print(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\n{'=' * 50}")
    except Exception as e:
        print(f"Stage failed: {e}")
        raise e
