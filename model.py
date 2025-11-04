import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitClassifier(nn.Module):
    """
    A simple neural network for recognizing single digits (0-9) from images.
    The network uses convolutional layers followed by fully connected layers.
    """
    
    def __init__(self):
        super(DigitClassifier, self).__init__()
        
        # First convolutional block
        # Input: 1 channel (grayscale), Output: 32 feature maps
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Max pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        # After conv2 and pooling twice: 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes for digits 0-9
        
    def forward(self, x):
        # First conv block: 28x28 -> 28x28 -> 14x14 (after pooling)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        
        # Second conv block: 14x14 -> 14x14 -> 7x7 (after pooling)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x


