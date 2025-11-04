# Digit Recognition Neural Network

A simple neural network example for recognizing single digits (0-9) from images using the MNIST dataset.

## Overview

This project implements a convolutional neural network (CNN) that can recognize handwritten digits from 28x28 grayscale images. The model uses:
- **Two convolutional layers** for feature extraction
- **Max pooling** for dimensionality reduction
- **Dropout** for regularization
- **Fully connected layers** for classification

## Files

- `model.py` - Neural network model definition
- `train.py` - Training script for the model
- `predict.py` - Prediction/inference script
- `requirements.txt` - Python dependencies

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Train the model on the MNIST dataset:
```bash
python train.py
```

This will:
- Download the MNIST dataset automatically
- Train the model for 10 epochs
- Save the trained model as `digit_classifier.pth`
- Generate `training_history.png` with training curves

### Testing/Predictions

#### Test on MNIST samples:
```bash
python predict.py
```
This will test the model on 10 random samples from the MNIST test set and save results as `test_predictions.png`.

#### Predict from your own image:
```bash
python predict.py path/to/your/image.png
```

**Important**: Your image should be:
- Grayscale (or will be converted automatically)
- Preferably 28x28 pixels (will be resized if needed)
- White digit on black background (inverse of MNIST style works too)

## Model Architecture

```
Input (1x28x28)
    ↓
Conv2d(1→32, kernel=3, padding=1) + ReLU
    ↓
MaxPool2d(2x2)
    ↓
Conv2d(32→64, kernel=3, padding=1) + ReLU
    ↓
MaxPool2d(2x2)
    ↓
Flatten (64x7x7 = 3136)
    ↓
Linear(3136→128) + ReLU + Dropout
    ↓
Linear(128→10)
    ↓
Output (10 classes: 0-9)
```

## Expected Performance

With the default settings, the model typically achieves:
- **Training accuracy**: ~98-99%
- **Test accuracy**: ~98-99%

Training takes approximately 5-10 minutes on a CPU, or 1-2 minutes on a GPU.

## Notes

- The model is trained on the MNIST dataset, which has black digits on white backgrounds
- If your images have white digits on black backgrounds, the model should still work reasonably well
- The model expects 28x28 grayscale images for best results

## Example

After training, you can visualize predictions:
```python
from predict import test_on_mnist
test_on_mnist(num_samples=20)  # Test on 20 random images
```

