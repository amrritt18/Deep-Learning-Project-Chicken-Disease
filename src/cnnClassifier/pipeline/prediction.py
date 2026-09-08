from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        model_path = Path("artifacts/training/model.h5")

        model = load_model(model_path)

        test_image = image.load_img(
            self.filename,
            target_size=(224, 224)
        )

        test_image = image.img_to_array(test_image)

        test_image = test_image / 255.0

        test_image = np.expand_dims(
            test_image,
            axis=0
        )

        probabilities = model.predict(
            test_image,
            verbose=0
        )[0]

        predicted_class = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[predicted_class]
        )

        if predicted_class == 1:
            prediction = "Healthy"
        else:
            prediction = "Coccidiosis"

        return [{
            "prediction": prediction,
            "confidence": confidence
        }]