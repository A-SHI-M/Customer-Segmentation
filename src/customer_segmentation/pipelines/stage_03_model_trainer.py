from customer_segmentation.config.configuration import ConfigurationManager
from customer_segmentation.components.model_trainer import ModelTrainer

STAGE_NAME = "Model Training Stage"


class ModelTrainerPipeline:
    def __init__(self):
        pass

    def initiate_model_training(self):
        config = ConfigurationManager()
        trainer_config = config.get_model_trainer_config()
        trainer = ModelTrainer(config=trainer_config)
        trainer.run()


if __name__ == "__main__":
    try:
        print(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerPipeline()
        obj.initiate_model_training()
        print(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\n{'=' * 50}")
    except Exception as e:
        print(f"Stage failed: {e}")
        raise e
