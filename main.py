from dotenv import load_dotenv

from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
)
from cnnClassifier.pipeline.stage_02_prepare_base_model import (
    PrepareBaseModelTrainingPipeline,
)
from cnnClassifier.pipeline.stage_03_model_trainer import (
    ModelTrainingPipeline,
)
from cnnClassifier.pipeline.stage_04_evaluation import (
    EvaluationPipeline,
)


load_dotenv()


STAGE_NAME = "Data Ingestion stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")

    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()

    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")

except Exception as e:
    logger.exception(
        f"{STAGE_NAME} failed: {e}"
    )
    raise


STAGE_NAME = "Prepare Base Model"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")

    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()

    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")

except Exception as e:
    logger.exception(
        f"{STAGE_NAME} failed: {e}"
    )
    raise


STAGE_NAME = "Training"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")

    model_trainer = ModelTrainingPipeline()
    model_trainer.main()

    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")

except Exception as e:
    logger.exception(
        f"{STAGE_NAME} failed: {e}"
    )
    raise


STAGE_NAME = "Evaluation stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")

    model_evaluation = EvaluationPipeline()
    model_evaluation.main()

    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")

except Exception as e:
    logger.exception(
        f"{STAGE_NAME} failed: {e}"
    )
    raise