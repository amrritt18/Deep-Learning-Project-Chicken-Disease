from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from cnnClassifier import logger


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model = None
        self.full_model = None

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        logger.info(f"Model saved at: {path}")

    def get_base_model(self):
        self.model = VGG16(
            input_shape=tuple(self.config.params_image_size),
            weights=self.config.params_weights,
            include_top=self.config.params_include_top,
        )

        self.save_model(
            path=self.config.base_model_path,
            model=self.model,
        )

        logger.info("VGG16 base model created successfully.")

    @staticmethod
    def _prepare_full_model(
        model: tf.keras.Model,
        classes: int,
        freeze_all: bool,
        freeze_till: int | None,
        learning_rate: float,
    ):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False

        elif freeze_till is not None and freeze_till > 0:
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

            for layer in model.layers[-freeze_till:]:
                layer.trainable = True

        x = model.output

        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(
            256,
            activation="relu"
        )(x)

        x = layers.Dropout(0.5)(x)

        prediction = layers.Dense(
            classes,
            activation="softmax"
        )(x)

        full_model = Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        trainable_params = sum(
            tf.keras.backend.count_params(weight)
            for weight in full_model.trainable_weights
        )

        non_trainable_params = sum(
            tf.keras.backend.count_params(weight)
            for weight in full_model.non_trainable_weights
        )

        logger.info(
            f"Trainable parameters: {trainable_params}"
        )

        logger.info(
            f"Non-trainable parameters: {non_trainable_params}"
        )

        full_model.summary()

        return full_model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=False,
            freeze_till=4,
            learning_rate=self.config.params_learning_rate,
        )

        self.save_model(
            path=self.config.updated_base_model_path,
            model=self.full_model,
        )

        logger.info(
            "VGG16 fine-tuning model created successfully."
        )