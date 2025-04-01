# Deep Learning Practice - Image Classification

## Project Overview
This project is part of the Deep Learning Practice (DLP) course. The objective is to train a Convolutional Neural Network (CNN) model to classify images of animals and plants with the best F1 score.

## Dataset
The dataset consists of 2000 test images of flora and fauna. Each image is labeled with a number between 0 to 9, representing different classes.

## Technology Used
- Python
- PyTorch
- Torchvision
- Transformers (Hugging Face)
- NumPy & Pandas
- Google Colab & Kaggle GPU
- Matplotlib & Seaborn (for visualization)

## Concepts Applied
- Convolutional Neural Networks (CNN)
- Transfer Learning (using pre-trained models like ResNet and VGG)
- Data Augmentation & Normalization
- Model Evaluation using F1 Score
- Training and Fine-tuning Deep Learning Models

## Implementation Details

### 1. Data Preprocessing
- Load dataset from Kaggle.
- Apply transformations like resizing, normalization, and augmentation.

### 2. Model Selection
- Used pre-trained models like **ResNet18** and **VGG19-BN**.
- Fine-tuned the models on the dataset.

### 3. Training & Optimization
- Used **Adam Optimizer** and **CrossEntropy Loss**.
- Trained the model with **GPU acceleration** on Kaggle.
- Saved the best model based on **F1 score**.

### 4. Inference & Submission
- Predicted labels for test images.
- Saved predictions in a CSV file.
- Submitted the CSV file to Kaggle.

## Instructions for Running on Google Colab

1. Open Google Colab and upload the notebook.
2. Install necessary dependencies:
   ```python
   !pip install torch torchvision transformers evaluate
3. Train the model and generate predictions.

