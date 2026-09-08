from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json
from cnnClassifier import logger


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.valid_generator = None
        self.score = None
        self.y_true = None
        self.y_pred = None
        self.y_prob = None
        self.metrics = None

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = (
            tf.keras.preprocessing.image.ImageDataGenerator(
                **datagenerator_kwargs
            )
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            class_mode="categorical",
            **dataflow_kwargs
        )

        logger.info(
            f"Validation samples: {self.valid_generator.samples}"
        )

        logger.info(
            f"Class indices: {self.valid_generator.class_indices}"
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        model = self.load_model(
            self.config.path_of_model
        )

        logger.info(
            f"Model loaded from: {self.config.path_of_model}"
        )

        self._valid_generator()

        self.score = model.evaluate(
            self.valid_generator,
            steps=len(self.valid_generator),
            verbose=1
        )

        logger.info(
            f"Evaluation loss: {self.score[0]:.4f}"
        )

        logger.info(
            f"Evaluation accuracy: {self.score[1]:.4f}"
        )

        self.valid_generator.reset()

        self.y_prob = model.predict(
            self.valid_generator,
            steps=len(self.valid_generator),
            verbose=1
        )

        self.y_true = self.valid_generator.classes

        self.y_pred = np.argmax(
            self.y_prob,
            axis=1
        )

        class_names = [
            name
            for name, index in sorted(
                self.valid_generator.class_indices.items(),
                key=lambda item: item[1]
            )
        ]

        accuracy = accuracy_score(
            self.y_true,
            self.y_pred
        )

        cm = confusion_matrix(
            self.y_true,
            self.y_pred,
            labels=[0, 1]
        )

        coccidiosis_precision = precision_score(
            self.y_true,
            self.y_pred,
            pos_label=0,
            zero_division=0
        )

        coccidiosis_recall = recall_score(
            self.y_true,
            self.y_pred,
            pos_label=0,
            zero_division=0
        )

        coccidiosis_f1 = f1_score(
            self.y_true,
            self.y_pred,
            pos_label=0,
            zero_division=0
        )

        healthy_precision = precision_score(
            self.y_true,
            self.y_pred,
            pos_label=1,
            zero_division=0
        )

        healthy_recall = recall_score(
            self.y_true,
            self.y_pred,
            pos_label=1,
            zero_division=0
        )

        healthy_f1 = f1_score(
            self.y_true,
            self.y_pred,
            pos_label=1,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            self.y_true,
            self.y_prob[:, 0]
        )

        report = classification_report(
            self.y_true,
            self.y_pred,
            labels=[0, 1],
            target_names=class_names,
            zero_division=0
        )

        self.metrics = {
            "loss": float(self.score[0]),
            "accuracy": float(accuracy),
            "coccidiosis": {
                "precision": float(coccidiosis_precision),
                "recall": float(coccidiosis_recall),
                "f1_score": float(coccidiosis_f1)
            },
            "healthy": {
                "precision": float(healthy_precision),
                "recall": float(healthy_recall),
                "f1_score": float(healthy_f1)
            },
            "roc_auc": float(roc_auc),
            "confusion_matrix": cm.tolist()
        }

        logger.info(
            f"Coccidiosis Precision: "
            f"{coccidiosis_precision:.4f}"
        )

        logger.info(
            f"Coccidiosis Recall: "
            f"{coccidiosis_recall:.4f}"
        )

        logger.info(
            f"Coccidiosis F1-score: "
            f"{coccidiosis_f1:.4f}"
        )

        logger.info(
            f"Healthy Precision: "
            f"{healthy_precision:.4f}"
        )

        logger.info(
            f"Healthy Recall: "
            f"{healthy_recall:.4f}"
        )

        logger.info(
            f"Healthy F1-score: "
            f"{healthy_f1:.4f}"
        )

        logger.info(
            f"ROC-AUC: {roc_auc:.4f}"
        )

        logger.info(
            f"Confusion Matrix:\n{cm}"
        )

        logger.info(
            f"Classification Report:\n{report}"
        )

    def save_score(self):
        save_json(
            path=Path("scores.json"),
            data=self.metrics
        )

        logger.info(
            "Evaluation scores saved successfully."
        )