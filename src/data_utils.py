"""
Utility functions for data loading and preprocessing.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

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
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0], std=[1])
    ])

    # Create dataset
    if dataset == "brain_mri":
        from datasets.brain_MRI_dataset import BrainMRIDataset
        CustomDataset = BrainMRIDataset
    dataset = CustomDataset(root_dir, transform=transform)
    
    # Split dataset into training and validation sets
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader