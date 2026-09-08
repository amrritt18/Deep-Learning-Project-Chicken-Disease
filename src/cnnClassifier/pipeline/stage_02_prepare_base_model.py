from cnnClassifier import logger
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier.config.configuration import ConfigurationManager


STAGE_NAME = "Prepare Base Model stage"


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logger.info(
                f">>>>>> Stage {STAGE_NAME} started <<<<<<"
            )

            config = ConfigurationManager()

            prepare_base_model_config = (
                config.get_prepare_base_model_config()
            )

            prepare_base_model = PrepareBaseModel(
                config=prepare_base_model_config
            )

            prepare_base_model.get_base_model()
            prepare_base_model.update_base_model()

            logger.info(
                f">>>>>> Stage {STAGE_NAME} completed <<<<<<"
            )

        except Exception as e:
            logger.exception(
                f"{STAGE_NAME} failed: {e}"
            )
            raise