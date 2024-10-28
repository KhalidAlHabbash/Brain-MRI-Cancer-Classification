# 🧠 Brain Tumor MRI Classification

A deep learning web application that classifies brain MRI scans into four categories: No Tumor, Glioma, Meningioma, and Pituitary Tumor. The project uses a Convolutional Neural Network (CNN) trained on the Brain Tumor MRI Dataset from Kaggle.

## Features

- Web-based interface for easy image upload and classification
- Real-time prediction using a trained CNN model
- Support for common image formats
- Responsive design with an intuitive user interface
- High accuracy classification (97.25% on test set)

## Technology Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask (Python)
- **Deep Learning**: PyTorch
- **Data Processing**: PIL, torchvision
- **Other Libraries**: NumPy, Matplotlib, Seaborn

## Model Architecture

The CNN model consists of:
- 4 Convolutional layers with batch normalization
- MaxPooling layers
- Dropout for regularization
- Fully connected layers
- ReLU activation functions

## Performance

The model achieves:
- Training Accuracy: 99.91%
- Validation Accuracy: 96.76%
- Test Accuracy: 97.25%

## Installation

1. Clone the repository
2. Install required packages from Conda environment

## Usage

1. Start the Flask server
2. The application will automatically open in your default web browser at `http://127.0.0.1:5001/`
3. Upload a brain MRI scan image using the interface
4. Click "Classify" to get the prediction

## Dataset

The model was trained on the Brain Tumor MRI Dataset from Kaggle, which includes:
- Training and testing sets
- Four classes of MRI scans
- Data augmentation techniques for improved generalization

## Model Training

The model was trained for 40 epochs with:
- Adam optimizer
- Cross-entropy loss
- Batch size of 32
- Learning rate of 0.001
- Data augmentation (random flips, rotations)
