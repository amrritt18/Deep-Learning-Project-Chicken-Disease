from pathlib import Path

import tensorflow as tf

from cnnClassifier.entity.config_entity import TrainingConfig
from cnnClassifier import logger


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.train_generator = None
        self.valid_generator = None

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

        logger.info(
            f"Base model loaded from: "
            f"{self.config.updated_base_model_path}"
        )

    def train_valid_generator(self):
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

        if self.config.params_is_augmentation:
            train_datagenerator = (
                tf.keras.preprocessing.image.ImageDataGenerator(
                    rotation_range=40,
                    horizontal_flip=True,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    **datagenerator_kwargs
                )
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            class_mode="categorical",
            **dataflow_kwargs
        )

        logger.info(
            f"Training samples: {self.train_generator.samples}"
        )

        logger.info(
            f"Validation samples: {self.valid_generator.samples}"
        )

        logger.info(
            f"Class indices: {self.train_generator.class_indices}"
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)

        logger.info(
            f"Trained model saved at: {path}"
        )

    def train(self):
        self.steps_per_epoch = len(self.train_generator)
        self.validation_steps = len(self.valid_generator)

        logger.info(
            f"Steps per epoch: {self.steps_per_epoch}"
        )

        logger.info(
            f"Validation steps: {self.validation_steps}"
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config.trained_model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.valid_generator,
            validation_steps=self.validation_steps,
            callbacks=[
                early_stopping,
                reduce_lr,
                checkpoint
            ]
        )

        logger.info(
            "Training completed successfully."
        )