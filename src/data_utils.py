"""
Utility functions for data loading and preprocessing.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src import datasets

def create_split_loaders(dataset: str, root_dir: str, image_size: tuple, batch_size: int, val_split: float = 0.2):
    """
    Create training and validation data loaders.
    
    Args:
        dataset (str): Name of the dataset
        root_dir (str): Directory containing the dataset
        image_size (tuple): Size of the images (height, width)
        batch_size (int): Batch size for data loaders
        val_split (float): Fraction of data to use for validation
        
    Returns:
        tuple: Training and validation DataLoaders
    """

    # Create dataset
    # if dataset == "brain_mri":
    #     from src.datasets import BrainMRIDataset
    #     CustomDataset = BrainMRIDataset
    # dataset = CustomDataset(root_dir, image_size=image_size)
    
    dataset = getattr(dataset, dataset)(root_dir, image_size)

    # Split dataset into training and validation sets
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def create_split_loaders_semi_supervised(dataset: str, root_dir: str, image_size: tuple, batch_size: int, label_split: float = 0.2, val_split: float = 0.2):
    """
    Create training and validation data loaders for semi-supervised learning.
    
    Args:
        dataset (str): Name of the dataset
        root_dir (str): Directory containing the dataset
        image_size (tuple): Size of the images (height, width)
        batch_size (int): Batch size for data loaders
        val_split (float): Fraction of data to use for validation
        
    Returns:
        tuple: Training and validation DataLoaders
    """

    # Create dataset
    dataset = getattr(datasets, dataset)(root_dir, image_size)

    # Split dataset into train and validation sets
    train_size = int((1 - val_split) * len(dataset))
    unlabeled_train_size = int((1 - label_split) * train_size)
    labeled_train_size = train_size - unlabeled_train_size
    val_size = len(dataset) - train_size
    unlabeled_train_dataset, labeled_train_dataset, val_dataset = torch.utils.data.random_split(dataset, [unlabeled_train_size, labeled_train_size, val_size])

    # Create data loaders
    labeled_train_loader = DataLoader(labeled_train_dataset, batch_size=batch_size, shuffle=True)
    unlabeled_train_loader = DataLoader(unlabeled_train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return labeled_train_loader, unlabeled_train_loader, val_loader