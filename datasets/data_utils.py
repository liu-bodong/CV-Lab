"""
Simple data loading utilities for image segmentation tasks.
Provides essential functions for creating data loaders and managing data splits.
"""

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
from typing import Tuple, Optional, List

def create_split_loaders(
    dataset, 
    batch_size: int, 
    val_split: float = 0.2,
    test_split: float = 0.0,
    num_workers: int = 4,
    shuffle: bool = True,
    **kwargs
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create training, validation, and test data loaders from a dataset.
                 
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size for data loaders
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for test (0.0 for no test set)
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the data
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        tuple: Training, validation, and optional test DataLoaders
    """
    
    # Create splits
    total_size = len(dataset)
    val_size = int(val_split * total_size)
    test_size = int(test_split * total_size) if test_split > 0 else 0
    train_size = total_size - val_size - test_size
    
    # Create random splits
    indices = list(range(total_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:] if test_size > 0 else []
    
    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices) if test_indices else None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            **kwargs
        )
    
    print(f"Created data loaders:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    if test_dataset:
        print(f"  Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader


def create_semi_supervised_loaders(
    dataset,
    batch_size: int,
    labeled_ratio: float = 0.1,
    val_split: float = 0.2,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create semi-supervised data loaders with labeled/unlabeled splits.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size for data loaders
        labeled_ratio: Ratio of training data to use as labeled
        val_split: Fraction of data to use for validation
        num_workers: Number of worker processes for data loading
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        tuple: Labeled training, unlabeled training, and validation DataLoaders
    """
    
    # Create train/val split first
    total_size = len(dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size
    
    indices = list(range(total_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create labeled/unlabeled split within training data
    labeled_size = int(labeled_ratio * train_size)
    labeled_indices = train_indices[:labeled_size]
    unlabeled_indices = train_indices[labeled_size:]
    
    # Create subset datasets
    labeled_dataset = Subset(dataset, labeled_indices)
    unlabeled_dataset = Subset(dataset, unlabeled_indices)
    val_dataset = Subset(dataset, val_indices)
    
    # Create data loaders
    labeled_loader = DataLoader(
        labeled_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs
    )
    
    unlabeled_loader = DataLoader(
        unlabeled_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs
    )
    
    print(f"Created semi-supervised data loaders:")
    print(f"  Labeled training: {len(labeled_dataset)} samples ({labeled_ratio*100:.1f}%)")
    print(f"  Unlabeled training: {len(unlabeled_dataset)} samples ({(1-labeled_ratio)*100:.1f}%)")
    print(f"  Validation: {len(val_dataset)} samples")
    
    return labeled_loader, unlabeled_loader, val_loader


def create_balanced_splits(
    dataset,
    batch_size: int,
    val_split: float = 0.2,
    test_split: float = 0.0,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create balanced data splits ensuring equal representation across classes.
    Note: This is a simple implementation. For true balanced splits, 
    you may need to implement class-aware splitting based on your dataset.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size for data loaders
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for test
        num_workers: Number of worker processes for data loading
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        tuple: Training, validation, and optional test DataLoaders
    """
    
    # For now, use random splits. Implement class-aware splitting as needed
    return create_split_loaders(
        dataset=dataset,
        batch_size=batch_size,
        val_split=val_split,
        test_split=test_split,
        num_workers=num_workers,
        **kwargs
    )


def create_stratified_splits(
    dataset,
    batch_size: int,
    val_split: float = 0.2,
    test_split: float = 0.0,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create stratified data splits maintaining class distribution.
    Note: This is a simple implementation. For true stratified splits, 
    you may need to implement class-aware splitting based on your dataset.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size for data loaders
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for test
        num_workers: Number of worker processes for data loading
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        tuple: Training, validation, and optional test DataLoaders
    """
    
    # For now, use random splits. Implement class-aware splitting as needed
    return create_split_loaders(
        dataset=dataset,
        batch_size=batch_size,
        val_split=val_split,
        test_split=test_split,
        num_workers=num_workers,
        **kwargs
    )


def get_dataset_statistics(dataset, sample_size: int = 100):
    """
    Get basic statistics about the dataset.
    
    Args:
        dataset: PyTorch dataset
        sample_size: Number of samples to analyze
        
    Returns:
        dict: Dataset statistics
    """
    
    total_samples = len(dataset)
    sample_size = min(sample_size, total_samples)
    
    # Sample a few images to get statistics
    sample_images = []
    sample_masks = []
    
    for i in range(sample_size):
        try:
            sample = dataset[i]
            if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                image, mask = sample[0], sample[1]
            else:
                image, mask = sample, None
            
            sample_images.append(image)
            if mask is not None:
                sample_masks.append(mask)
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            continue
    
    statistics = {
        'total_samples': total_samples,
        'analyzed_samples': len(sample_images),
    }
    
    if sample_images:
        # Calculate image statistics
        if isinstance(sample_images[0], torch.Tensor):
            image_tensor = torch.stack(sample_images)
            image_mean = image_tensor.mean(dim=[0, 2, 3]).tolist()
            image_std = image_tensor.std(dim=[0, 2, 3]).tolist()
        else:
            # Handle PIL images or other formats
            image_mean = [0.0, 0.0, 0.0]  # Placeholder
            image_std = [1.0, 1.0, 1.0]   # Placeholder
        
        statistics.update({
            'image_mean': image_mean,
            'image_std': image_std,
        })
        
        # Calculate mask statistics if available
        if sample_masks:
            if isinstance(sample_masks[0], torch.Tensor):
                mask_tensor = torch.stack(sample_masks)
                mask_mean = mask_tensor.mean().item()
                mask_std = mask_tensor.std().item()
                
                # Calculate class distribution
                unique_values = torch.unique(mask_tensor)
                class_distribution = {int(val.item()): (mask_tensor == val).sum().item() 
                                    for val in unique_values}
            else:
                mask_mean = 0.0
                mask_std = 0.0
                class_distribution = {}
            
            statistics.update({
                'mask_mean': mask_mean,
                'mask_std': mask_std,
                'class_distribution': class_distribution,
                'num_classes': len(class_distribution),
            })
    
    return statistics


def create_two_stream_sampler(primary_indices, secondary_indices, batch_size, secondary_batch_size):
    """
    Create a two-stream batch sampler for semi-supervised learning.
    
    Args:
        primary_indices: Indices for primary stream (e.g., labeled data)
        secondary_indices: Indices for secondary stream (e.g., unlabeled data)
        batch_size: Total batch size
        secondary_batch_size: Number of samples from secondary stream
        
    Returns:
        TwoStreamBatchSampler instance
    """
    
    from .data_aug import TwoStreamBatchSampler
    return TwoStreamBatchSampler(
        primary_indices=primary_indices,
        secondary_indices=secondary_indices,
        batch_size=batch_size,
        secondary_batch_size=secondary_batch_size
    )


# Utility functions for augmentation configuration
def get_weak_augmentation_config():
    """Get configuration for weak augmentations."""
    return {
        'rotation': True,
        'rotation_degrees': 10,
        'flip': True,
        'flip_p': 0.3,
        'color_jitter': True,
        'brightness': 0.1,
        'contrast': 0.1,
        'saturation': 0.1,
        'hue': 0.05,
    }


def get_strong_augmentation_config():
    """Get configuration for strong augmentations."""
    return {
        'rotation': True,
        'rotation_degrees': 30,
        'flip': True,
        'flip_p': 0.5,
        'crop': True,
        'color_jitter': True,
        'brightness': 0.3,
        'contrast': 0.3,
        'saturation': 0.3,
        'hue': 0.1,
        'blur': True,
        'blur_p': 0.3,
        'elastic': True,
        'elastic_p': 0.2,
        'grid_distortion': True,
        'grid_distortion_p': 0.2,
        'brightness_contrast': True,
        'gamma': True,
    }


def get_medical_augmentation_config():
    """Get configuration for medical image augmentations."""
    return {
        'rotation': True,
        'rotation_degrees': 15,
        'flip': True,
        'flip_p': 0.4,
        'color_jitter': True,
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.1,
        'hue': 0.05,
        'blur': True,
        'blur_p': 0.2,
        'brightness_contrast': True,
    }