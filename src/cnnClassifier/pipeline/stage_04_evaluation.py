from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.evaluation import Evaluation
from cnnClassifier import logger


STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logger.info(
                f">>>>>> Stage {STAGE_NAME} started <<<<<<"
            )

            config = ConfigurationManager()

            val_config = config.get_validation_config()

            evaluation = Evaluation(
                config=val_config
            )

            evaluation.evaluation()
            evaluation.save_score()

            logger.info(
                f">>>>>> Stage {STAGE_NAME} completed <<<<<<"
            )

        except Exception as e:
            logger.exception(
                f"{STAGE_NAME} failed: {e}"
            )
            raise