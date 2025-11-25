"""
CNN Models for Malware Detection
Implements the 3C2D CNN architecture from the paper for both pipelines
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class C3C2D_SingleChannel(nn.Module):
    """
    Pipeline 1: 3C2D CNN for single-channel bigram-DCT images
    
    Architecture (exactly as defined in paper):
    - Input: 256×256×1
    - Conv(32, 3×3) → MaxPool(2×2)
    - Conv(64, 3×3) → MaxPool(2×2)
    - Conv(128, 3×3) → MaxPool(2×2)
    - Dense(512) + Dropout(0.5)
    - Dense(256) + Dropout(0.5)
    - Dense(1, activation=sigmoid)
    """
    
    def __init__(self):
        super(C3C2D_SingleChannel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        # Input: 256×256
        # After pool1: 128×128
        # After pool2: 64×64
        # After pool3: 32×32
        # Flattened: 128 * 32 * 32 = 131,072
        
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(256, 1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 1, 256, 256)
            
        Returns:
            Output tensor of shape (batch_size, 1) with sigmoid activation
        """
        # Conv block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten
        x = self.flatten(x)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = torch.sigmoid(x)
        
        return x


class C3C2D_TwoChannel(nn.Module):
    """
    Pipeline 2: 3C2D CNN for two-channel ensemble images (byteplot + bigram-DCT)
    
    Architecture (same as Pipeline 1, but with 2-channel input):
    - Input: 256×256×2
    - Conv(32, 3×3) → MaxPool(2×2)
    - Conv(64, 3×3) → MaxPool(2×2)
    - Conv(128, 3×3) → MaxPool(2×2)
    - Dense(512) + Dropout(0.5)
    - Dense(256) + Dropout(0.5)
    - Dense(1, activation=sigmoid)
    """
    
    def __init__(self):
        super(C3C2D_TwoChannel, self).__init__()
        
        # Convolutional layers (only difference: in_channels=2 for first conv)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Flatten
        self.flatten = nn.Flatten()
        
        # Fully connected layers (same as single-channel)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(256, 1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 2, 256, 256)
            
        Returns:
            Output tensor of shape (batch_size, 1) with sigmoid activation
        """
        # Conv block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten
        x = self.flatten(x)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = torch.sigmoid(x)
        
        return x


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the models
    print("Testing Pipeline 1 (Single-Channel) Model...")
    model1 = C3C2D_SingleChannel()
    print(f"Total parameters: {count_parameters(model1):,}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 1, 256, 256)
    output = model1(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    print("\n" + "="*60 + "\n")
    
    print("Testing Pipeline 2 (Two-Channel) Model...")
    model2 = C3C2D_TwoChannel()
    print(f"Total parameters: {count_parameters(model2):,}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 2, 256, 256)
    output = model2(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    print("\nModel architectures:")
    print("\nPipeline 1:")
    print(model1)
    print("\nPipeline 2:")
    print(model2)
