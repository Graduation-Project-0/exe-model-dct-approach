"""
PyTorch Dataset and DataLoader for Malware Detection
Loads executable files and generates images for both pipelines
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Optional
import random


class MalwareImageDataset(Dataset):
    """
    Dataset for loading executable files and generating images.
    Supports both Pipeline 1 (single-channel bigram-DCT) and Pipeline 2 (two-channel ensemble).
    """
    
    def __init__(
        self, 
        data_dir: str, 
        mode: str = 'bigram_dct',  # 'bigram_dct' or 'two_channel'
        max_samples: Optional[int] = None
    ):
        """
        Args:
            data_dir: Root directory containing 'malware' and 'benign' subdirectories
            mode: 'bigram_dct' for Pipeline 1, 'two_channel' for Pipeline 2
            max_samples: Maximum number of samples to load (None for all)
        """
        self.data_dir = data_dir
        self.mode = mode
        self.samples = []  # List of (file_path, label) tuples
        
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.image_generation import create_bigram_dct_image, create_two_channel_image
        
        self.create_bigram_dct_image = create_bigram_dct_image
        self.create_two_channel_image = create_two_channel_image
        
        self._load_samples(max_samples)
        
    def _load_samples(self, max_samples: Optional[int]):
        """Load all executable file paths with their labels."""
        malware_dir = os.path.join(self.data_dir, 'malware')
        benign_dir = os.path.join(self.data_dir, 'benign')
        
        # Load malware samples (label = 1)
        if os.path.exists(malware_dir):
            for filename in os.listdir(malware_dir):
                file_path = os.path.join(malware_dir, filename)
                if os.path.isfile(file_path):
                    self.samples.append((file_path, 1))
        
        # Load benign samples (label = 0)
        if os.path.exists(benign_dir):
            for filename in os.listdir(benign_dir):
                file_path = os.path.join(benign_dir, filename)
                if os.path.isfile(file_path):
                    self.samples.append((file_path, 0))
        
        random.shuffle(self.samples)
        
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples)} samples")
        malware_count = sum(1 for _, label in self.samples if label == 1)
        benign_count = len(self.samples) - malware_count
        print(f"  Malware: {malware_count}, Benign: {benign_count}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load and process a sample.
        
        Returns:
            image: Tensor of shape (C, H, W) where C=1 for bigram_dct, C=2 for two_channel
            label: 0 for benign, 1 for malware
        """
        file_path, label = self.samples[idx]
        
        try:
            if self.mode == 'bigram_dct':
                # Pipeline 1: Single-channel bigram-DCT
                image = self.create_bigram_dct_image(file_path)
                # Add channel dimension: (H, W) -> (1, H, W)
                image = np.expand_dims(image, axis=0)
            
            elif self.mode == 'two_channel':
                # Pipeline 2: Two-channel ensemble
                image = self.create_two_channel_image(file_path)
                # Convert (H, W, C) -> (C, H, W)
                image = np.transpose(image, (2, 0, 1))
            
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
            
            # Convert to float32 tensor
            image_tensor = torch.from_numpy(image).float()
            
            return image_tensor, label
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Return a zero image if processing fails
            if self.mode == 'bigram_dct':
                image_tensor = torch.zeros((1, 256, 256), dtype=torch.float32)
            else:
                image_tensor = torch.zeros((2, 256, 256), dtype=torch.float32)
            return image_tensor, label


def create_data_loaders(
    data_dir: str,
    mode: str = 'bigram_dct',
    batch_size: int = 32,
    train_split: float = 0.7,
    val_split: float = 0.2,
    test_split: float = 0.1,
    max_samples: Optional[int] = None,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir: Root directory with malware/benign subdirectories
        mode: 'bigram_dct' or 'two_channel'
        batch_size: Batch size for training
        train_split: Fraction of data for training (0.7 = 70%)
        val_split: Fraction of data for validation (0.2 = 20%)
        test_split: Fraction of data for testing (0.1 = 10%)
        max_samples: Maximum samples to load (None for all)
        num_workers: Number of worker processes for data loading
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create full dataset
    full_dataset = MalwareImageDataset(data_dir, mode=mode, max_samples=max_samples)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"\nDataset splits:")
    print(f"  Train: {train_size} ({train_split*100:.0f}%)")
    print(f"  Val:   {val_size} ({val_split*100:.0f}%)")
    print(f"  Test:  {test_size} ({test_split*100:.0f}%)")
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    data_dir = "./data"
    
    print("Testing Pipeline 1 (Bigram-DCT)...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir,
        mode='bigram_dct',
        batch_size=8,
        max_samples=100
    )
    
    # Test loading a batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}, Labels: {labels.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        break
    
    print("\nTesting Pipeline 2 (Two-Channel)...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir,
        mode='two_channel',
        batch_size=8,
        max_samples=100
    )
    
    # Test loading a batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}, Labels: {labels.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        break
