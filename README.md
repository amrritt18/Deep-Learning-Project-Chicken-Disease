# Chicken Fecal Disease Classification Using Deep Learning

A deep learning project for classifying chicken fecal images into **Coccidiosis** and **Healthy** classes using transfer learning with **VGG16**.

The project follows a modular machine-learning pipeline covering data ingestion, model preparation, training, evaluation, and deployment with Streamlit.

## Project Overview

Coccidiosis is a common parasitic disease in poultry that can affect chicken health and productivity. Early identification can help poultry professionals take appropriate action.

This project develops an image classification system that analyzes chicken fecal images and predicts whether the sample is:

- **Coccidiosis**
- **Healthy**

A pretrained VGG16 convolutional neural network is used as the feature extractor, followed by fine-tuning of the final convolutional block for the chicken fecal image classification task.

## Features

- Transfer learning using pretrained VGG16
- Fine-tuning of VGG16 Block 5
- Image augmentation during training
- Modular machine-learning pipeline
- YAML-based configuration
- Model checkpointing
- Early stopping
- Learning-rate reduction
- Class-wise evaluation
- Confusion matrix
- Precision, Recall, and F1-score
- ROC-AUC evaluation
- Streamlit web application for inference
- AWS S3-compatible data ingestion structure

## Dataset

The dataset contains chicken fecal images belonging to two classes:

```text
Chicken-fecal-images/
├── Coccidiosis/
└── Healthy/
```

### Dataset Statistics

| Property | Value |
|---|---:|
| Total Images | 390 |
| Training Images | 312 |
| Validation Images | 78 |
| Number of Classes | 2 |
| Input Image Size | 224 × 224 |
| Validation Split | 20% |

The class mapping used by Keras is:

```text
Coccidiosis → 0
Healthy → 1
```

## Model Architecture

The project uses **VGG16 pretrained on ImageNet** as the convolutional feature extractor.

The classification architecture is:

```text
Input Image
    ↓
VGG16 Backbone
    ↓
Global Average Pooling
    ↓
Dense Layer (256, ReLU)
    ↓
Dropout (0.5)
    ↓
Dense Layer (2, Softmax)
    ↓
Coccidiosis / Healthy
```

## Transfer Learning Experiment

The first experiment used a frozen VGG16 backbone, with only the newly added classification layers trained.

### Frozen VGG16 Results

- Validation Accuracy: **92.31%**
- ROC-AUC: **0.9645**
- Confusion Matrix:

```text
[[35, 4],
 [ 2,37]]
```

This provided the baseline for the next experiment.

## Fine-Tuning Experiment

In the second experiment, the final VGG16 convolutional block (**Block 5**) was unfrozen so that deeper pretrained features could adapt to chicken fecal image characteristics.

A lower learning rate was used for fine-tuning to reduce the risk of destroying useful pretrained features.

### Fine-Tuned VGG16 Results

- Validation Accuracy: **94.87%**
- Coccidiosis Precision: **97.30%**
- Coccidiosis Recall: **92.31%**
- Coccidiosis F1-score: **94.81%**
- Healthy Precision: **92.68%**
- Healthy Recall: **97.44%**
- Healthy F1-score: **95.06%**

Confusion matrix:

```text
                 Predicted
              Coccidiosis  Healthy

Actual
Coccidiosis       36          3
Healthy            1         38
```

The fine-tuned model was selected as the final model because it improved validation performance over the frozen VGG16 baseline.

## Experiment Comparison

| Experiment | Trainable Strategy | Validation Accuracy |
|---|---|---:|
| Experiment 1 | Frozen VGG16 | 92.31% |
| Experiment 2 | Fine-tuned VGG16 Block 5 | **94.87%** |

Fine-tuning improved validation accuracy by approximately **2.56 percentage points**.

It also improved Coccidiosis recall from **89.74% to 92.31%**, which is important because correctly identifying diseased samples is a key objective of the classifier.

## Training Configuration

The main training parameters are stored in `params.yaml`.

```yaml
AUGMENTATION: True
IMAGE_SIZE: [224, 224, 3]
BATCH_SIZE: 16
INCLUDE_TOP: False
EPOCHS: 20
CLASSES: 2
WEIGHTS: imagenet
LEARNING_RATE: 0.00001
```

The training pipeline uses:

- Adam optimizer
- Categorical cross-entropy loss
- Batch size: 16
- Maximum epochs: 20
- EarlyStopping
- ReduceLROnPlateau
- Best-model checkpointing

## Data Augmentation

When augmentation is enabled, the training pipeline applies:

- Rotation
- Horizontal flipping
- Width shifting
- Height shifting
- Shearing
- Zooming

Validation images are not augmented.

## Image Preprocessing

Images are:

1. Converted to RGB
2. Resized to `224 × 224`
3. Converted to NumPy arrays
4. Normalized by dividing pixel values by 255

```python
image = image / 255.0
```

The same preprocessing is used during inference to maintain consistency between training and deployment.

## Evaluation

The evaluation pipeline calculates:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Classification report

### Final Classification Report

```text
              precision    recall  f1-score   support

Coccidiosis       0.97      0.92      0.95        39
Healthy           0.93      0.97      0.95        39

accuracy                              0.95        78
macro avg         0.95      0.95      0.95        78
weighted avg      0.95      0.95      0.95        78
```

The validation set contains 78 images, with 39 images from each class.

## Project Structure

```text
Deep-Learning-Project-Chicken-Disease/
│
├── app.py
├── main.py
├── params.yaml
├── pyproject.toml
├── README.md
├── .gitignore
│
├── config/
│   └── config.yaml
│
├── research/
│   └── trials.ipynb
│
├── templates/
│   └── index.html
│
└── src/
    └── cnnClassifier/
        ├── __init__.py
        │
        ├── components/
        │   ├── data_ingestion.py
        │   ├── prepare_base_model.py
        │   ├── model_trainer.py
        │   └── evaluation.py
        │
        ├── config/
        │   └── configuration.py
        │
        ├── constants/
        │   └── __init__.py
        │
        ├── entity/
        │   └── config_entity.py
        │
        ├── pipeline/
        │   ├── stage_01_data_ingestion.py
        │   ├── stage_02_prepare_base_model.py
        │   ├── stage_03_model_trainer.py
        │   └── stage_04_evaluation.py
        │
        └── utils/
            └── common.py
```

## Pipeline

The complete workflow is:

```text
Dataset
   ↓
Data Ingestion
   ↓
Base Model Preparation
   ↓
Transfer Learning / Fine-Tuning
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Selection
   ↓
Streamlit Deployment
```

### Stage 1 — Data Ingestion

The project includes a data-ingestion component designed to support downloading and preparing the dataset from Amazon S3.

### Stage 2 — Prepare Base Model

Loads pretrained VGG16 weights and constructs the custom classification head.

For the fine-tuning experiment, VGG16 Block 5 is made trainable while earlier layers remain frozen.

### Stage 3 — Model Training

Creates training and validation generators and trains the model using:

- Data augmentation
- Early stopping
- Learning-rate reduction
- Best-model checkpointing

### Stage 4 — Evaluation

Loads the best saved model and evaluates it on the validation split using classification and disease-specific metrics.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/amrritt18/Deep-Learning-Project-Chicken-Disease.git
cd Deep-Learning-Project-Chicken-Disease
```

### 2. Install dependencies

This project uses `uv`.

```bash
uv sync
```

### 3. Activate the virtual environment

```bash
source .venv/bin/activate
```

## Dataset Setup

Place the dataset in:

```text
data/Chicken-fecal-images/
```

with the following structure:

```text
data/
└── Chicken-fecal-images/
    ├── Coccidiosis/
    └── Healthy/
```

## Running the Training Pipeline

### Stage 2 — Prepare Base Model

```bash
uv run python -c "from src.cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline; PrepareBaseModelTrainingPipeline().main()"
```

### Stage 3 — Train Model

```bash
uv run python -c "from src.cnnClassifier.pipeline.stage_03_model_trainer import ModelTrainingPipeline; ModelTrainingPipeline().main()"
```

The best checkpoint is saved locally under:

```text
artifacts/training/model.h5
```

The trained model artifacts are intentionally excluded from GitHub because the `.h5` files exceed GitHub's normal file-size limits.

### Stage 4 — Evaluate Model

```bash
uv run python -c "from src.cnnClassifier.pipeline.stage_04_evaluation import EvaluationPipeline; EvaluationPipeline().main()"
```

Evaluation results are saved to:

```text
scores.json
```

## Streamlit Application

The project includes a Streamlit application for image-based inference.

The application allows users to:

1. Upload a chicken fecal image
2. Preview the uploaded image
3. Run the trained model
4. View the predicted class
5. View prediction confidence
6. View class probabilities

### Install Streamlit

```bash
uv add streamlit
```

### Run the Application

```bash
uv run streamlit run app.py
```

The application uses the fine-tuned model:

```text
artifacts/training/vgg16_finetuned.h5
```

## Technologies Used

- Python 3.11
- TensorFlow 2.12
- Keras
- VGG16
- NumPy
- Scikit-learn
- Pillow
- PyYAML
- Python-Box
- Joblib
- Streamlit
- AWS S3
- Git
- GitHub
- uv

## Why VGG16?

VGG16 was selected as the initial transfer-learning architecture because:

- It is a well-established CNN architecture.
- It has strong pretrained ImageNet features.
- It is straightforward to fine-tune.
- It provides a useful baseline for a small image dataset.
- Its architecture is relatively easy to explain during technical interviews.

The dataset contains only 390 images. Training a deep CNN entirely from scratch could increase the risk of overfitting. Transfer learning allows the model to reuse visual features learned from ImageNet and adapt them to the target classification problem.

## Why Fine-Tune Block 5?

The first experiment froze the complete VGG16 backbone and achieved 92.31% validation accuracy.

The second experiment unfroze the final VGG16 block and used a lower learning rate. This allowed the deeper feature representations to adapt to the visual characteristics of chicken fecal images while retaining the useful low-level and mid-level features learned from ImageNet.

The resulting validation accuracy increased to 94.87%.

## Limitations

This project has several limitations:

- The dataset is relatively small.
- The evaluation currently uses a validation split rather than a completely independent test set.
- Performance may vary on images captured using different cameras, lighting conditions, backgrounds, or acquisition environments.
- The model should not be considered a replacement for professional veterinary diagnosis.
- The reported metrics are based on a relatively small validation set and therefore should be interpreted with appropriate caution.

For a production-grade system, a larger and more diverse dataset and a completely independent test set would be desirable.

## Future Improvements

Possible future improvements include:

- Collecting more chicken fecal images
- Creating a completely independent test dataset
- Evaluating additional CNN architectures
- Comparing VGG16 with ResNet and EfficientNet
- Hyperparameter optimization
- Class imbalance analysis as the dataset grows
- Explainable AI using Grad-CAM
- Confidence thresholding
- AWS S3-based dataset and model management
- Docker containerization
- Cloud deployment
- Model performance monitoring
- Continuous model retraining with new data

## Important Note

This project is intended as an **AI/ML research and educational project** demonstrating image classification, transfer learning, fine-tuning, evaluation, and deployment.

Predictions should not be used as a substitute for veterinary examination or laboratory diagnosis.

## Author

**Amrritt**

AI/ML Student

GitHub: https://github.com/amrritt18
