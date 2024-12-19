from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_trainer import Training
from cnnClassifier import logger
import pathlib as Path

STAGE_NAME = "Model trainer"

class ModelTrainingTrainingPipeline:
    def __init__(self):
      pass

    def main(self):
       config =ConfigurationManager()
       training_config = config.get_training_config()
       training = Training(config=training_config)
       training.get_base_model()
       training.train_valid_generator()
       training.train()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

