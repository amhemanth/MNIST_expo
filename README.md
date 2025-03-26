# MNIST Model with Specific Requirements

[![Model Requirements Validation](https://github.com/amhemanth/MNIST_expo/actions/workflows/model_validation.yml/badge.svg)](https://github.com/amhemanth/MNIST_expo/actions/workflows/model_validation.yml)

This project implements a PyTorch model for MNIST classification that meets the following requirements:

- Achieves 99.4% validation/test accuracy (using 50k/10k split)
- Uses less than 20k parameters
- Trains in less than 20 epochs
- Incorporates Batch Normalization and Dropout
- Uses either Fully Connected layer or Global Average Pooling

## Model Architecture

```
Input (1x28x28)
     ↓
[Conv2d(k=3, p=1) → BN → ReLU]  →  8x28x28
     ↓
[Conv2d(k=3, p=1) → BN → ReLU]  →  16x28x28
     ↓
[MaxPool2d(2)]  →  16x14x14
     ↓
[Dropout(0.1)]
     ↓
[Conv2d(k=3, p=1) → BN → ReLU]  →  16x14x14
     ↓
[Global Avg Pool]  →  16x1x1
     ↓
[Flatten]  →  16
     ↓
[Linear]  →  10
     ↓
[LogSoftmax]
     ↓
Output (10 classes)
```

Key Features:
- 3 convolutional layers with batch normalization
- Dropout for regularization (10%)
- Global Average Pooling
- Single fully connected layer
- Total parameters: < 20k

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (optional but recommended):
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Training

To train the model:

```bash
python model.py
```

The script will:
1. Download the MNIST dataset automatically
2. Split the training data into 50k/10k train/validation sets
3. Train for 15 epochs
4. Save the best model based on validation accuracy
5. Print the final test accuracy and parameter count

Expected output:
```
Train Epoch: 1 [...]
...
Final Test Accuracy: ~99.4%
Total number of parameters: <20k
```

## Model Details

Layer-wise dimensions:
- Input: 1x28x28 (MNIST image)
- Conv1: 8x28x28 (8 channels, spatial dim preserved with padding)
- Conv2: 16x28x28 (16 channels, spatial dim preserved with padding)
- MaxPool: 16x14x14 (spatial dim halved)
- Conv3: 16x14x14 (channels unchanged, spatial dim preserved)
- GAP: 16x1x1 (spatial dims collapsed)
- FC: 10 (output classes)

## Validation

GitHub Actions automatically validates that the model meets all requirements:
- Parameter count < 20k
- Use of Batch Normalization
- Use of Dropout
- Use of FC layer or GAP

The validation workflow runs on every push and pull request. 