# Simple Dataset and Data Loading Utilities

This module provides essential data loading utilities for image segmentation tasks. Users are responsible for implementing their own datasets and augmentations.

## Overview

The system is designed to be:
- **Simple**: Easy-to-use functions without complex abstractions
- **Flexible**: Works with any PyTorch dataset
- **Focused**: Provides only essential utilities
- **User-Responsible**: Users implement their own datasets and augmentations

## Quick Start

### Basic Usage

```python
from datasets import create_split_loaders

# Create your own dataset
class MyDataset:
    def __init__(self, root_dir, augmentations=None):
        self.augmentations = augmentations
        # ... implement your data loading logic
    
    def __getitem__(self, idx):
        image, mask = # ... load your data
        
        # Apply your own augmentations if specified
        if self.augmentations:
            image, mask = self.augmentations(image, mask)
        
        return image, mask

# Create dataset with your own augmentations
def my_augmentations(image, mask):
    # Implement your own augmentation logic
    import random
    from torchvision.transforms.functional import hflip
    
    if random.random() < 0.5:
        image = hflip(image)
        mask = hflip(mask)
    
    return image, mask

dataset = MyDataset(
    root_dir="./data/my_dataset",
    augmentations=my_augmentations
)

# Create data loaders
train_loader, val_loader, test_loader = create_split_loaders(
    dataset=dataset,
    batch_size=8,
    val_split=0.2
)
```

### Semi-Supervised Learning

```python
from datasets import create_semi_supervised_loaders

# Create semi-supervised data loaders
labeled_loader, unlabeled_loader, val_loader = create_semi_supervised_loaders(
    dataset=dataset,
    batch_size=8,
    labeled_ratio=0.1,  # 10% labeled data
    val_split=0.2
)
```

## Available Functions

### Data Loading Utilities

- `create_split_loaders()`: Create train/val/test data loaders
- `create_semi_supervised_loaders()`: Create labeled/unlabeled/val data loaders
- `get_dataset_statistics()`: Get basic dataset statistics
- `create_balanced_splits()`: Create balanced data splits (placeholder)
- `create_stratified_splits()`: Create stratified data splits (placeholder)

### Configuration Functions

- `get_weak_augmentation_config()`: Get weak augmentation configuration (reference)
- `get_strong_augmentation_config()`: Get strong augmentation configuration (reference)
- `get_medical_augmentation_config()`: Get medical augmentation configuration (reference)

**Note**: These configuration functions provide reference implementations. Users implement their own augmentation functions based on these configs or create completely custom ones.

## Usage Examples

### Example 1: Basic Dataset with Custom Augmentations

```python
import torch
from PIL import Image
import torchvision.transforms as transforms
import random
from torchvision.transforms.functional import hflip, vflip, rotate

class MySegmentationDataset:
    def __init__(self, root_dir, image_size=(256, 256), augmentations=None):
        self.root_dir = root_dir
        self.image_size = image_size
        self.augmentations = augmentations
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        
        # Load your data pairs here
        self.data_pairs = self._load_data()
    
    def _load_data(self):
        # Implement your data loading logic
        # Return list of (image_path, mask_path) tuples
        pass
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        image_path, mask_path = self.data_pairs[idx]
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Apply your own augmentations
        if self.augmentations:
            image, mask = self.augmentations(image, mask)
        
        # Apply transforms
        image = self.transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()
        
        return image, mask

# Define your own augmentation function
def my_augmentations(image, mask):
    # Random horizontal flip
    if random.random() < 0.5:
        image = hflip(image)
        mask = hflip(mask)
    
    # Random rotation
    if random.random() < 0.3:
        angle = random.uniform(-15, 15)
        image = rotate(image, angle)
        mask = rotate(mask, angle)
    
    return image, mask

# Use the dataset
dataset = MySegmentationDataset(
    root_dir="./data/my_dataset",
    augmentations=my_augmentations
)
```

### Example 2: Using Configuration References

```python
from datasets import get_weak_augmentation_config, get_strong_augmentation_config

# Get reference configurations
weak_config = get_weak_augmentation_config()
strong_config = get_strong_augmentation_config()

print("Weak augmentation config:", weak_config)
print("Strong augmentation config:", strong_config)

# Implement your own augmentations based on these configs
def weak_augmentations(image, mask):
    """Weak augmentations for labeled data."""
    import random
    from torchvision.transforms.functional import hflip, rotate
    
    # Use the config as reference
    if random.random() < weak_config['flip_p']:
        image = hflip(image)
        mask = hflip(mask)
    
    if random.random() < 0.3:
        angle = random.uniform(-weak_config['rotation_degrees'], weak_config['rotation_degrees'])
        image = rotate(image, angle)
        mask = rotate(mask, angle)
    
    return image, mask

def strong_augmentations(image, mask):
    """Strong augmentations for unlabeled data."""
    import random
    from torchvision.transforms.functional import hflip, vflip, rotate
    
    # Use the config as reference
    if random.random() < strong_config['flip_p']:
        image = hflip(image)
        mask = hflip(mask)
    
    if random.random() < 0.3:
        image = vflip(image)
        mask = vflip(mask)
    
    if random.random() < 0.5:
        angle = random.uniform(-strong_config['rotation_degrees'], strong_config['rotation_degrees'])
        image = rotate(image, angle)
        mask = rotate(mask, angle)
    
    return image, mask
```

### Example 3: Semi-Supervised Learning

```python
from datasets import create_semi_supervised_loaders

# Define different augmentation strategies
def weak_augmentations(image, mask):
    """Weak augmentations for labeled data."""
    import random
    from torchvision.transforms.functional import hflip
    
    if random.random() < 0.3:
        image = hflip(image)
        mask = hflip(mask)
    
    return image, mask

def strong_augmentations(image, mask):
    """Strong augmentations for unlabeled data."""
    import random
    from torchvision.transforms.functional import hflip, vflip, rotate
    
    if random.random() < 0.5:
        image = hflip(image)
        mask = hflip(mask)
    
    if random.random() < 0.3:
        image = vflip(image)
        mask = vflip(mask)
    
    if random.random() < 0.5:
        angle = random.uniform(-30, 30)
        image = rotate(image, angle)
        mask = rotate(mask, angle)
    
    return image, mask

# Create datasets with different augmentation strategies
labeled_dataset = MySegmentationDataset(
    root_dir="./data/my_dataset",
    augmentations=weak_augmentations  # Weak for labeled data
)

unlabeled_dataset = MySegmentationDataset(
    root_dir="./data/my_dataset",
    augmentations=strong_augmentations  # Strong for unlabeled data
)

# Create semi-supervised loaders
labeled_loader, unlabeled_loader, val_loader = create_semi_supervised_loaders(
    dataset=labeled_dataset,  # Use labeled dataset as base
    batch_size=8,
    labeled_ratio=0.1,  # 10% labeled data
    val_split=0.2
)
```

### Example 4: Different Augmentation Strategies

```python
from datasets import get_medical_augmentation_config

# Get medical augmentation config as reference
medical_config = get_medical_augmentation_config()

def medical_augmentations(image, mask):
    """Medical image specific augmentations."""
    import random
    from torchvision.transforms.functional import hflip, rotate
    
    # Use medical config as reference
    if random.random() < medical_config['flip_p']:
        image = hflip(image)
        mask = hflip(mask)
    
    if random.random() < 0.4:
        angle = random.uniform(-medical_config['rotation_degrees'], medical_config['rotation_degrees'])
        image = rotate(image, angle)
        mask = rotate(mask, angle)
    
    return image, mask

# Create medical dataset
medical_dataset = MySegmentationDataset(
    root_dir="./data/my_dataset",
    augmentations=medical_augmentations
)
```

## Configuration Examples

### YAML Configuration

```yaml
# hyper.yaml
dataset: MySegmentationDataset
data_dir: ./data/my_dataset
image_size: [256, 256]
batch_size: 8
val_split: 0.2

# Your custom augmentation configuration
augmentations:
  rotation: true
  rotation_degrees: 15
  flip: true
  flip_p: 0.5
  color_jitter: true
  brightness: 0.2
  contrast: 0.2

# Semi-supervised configuration
semi_supervised:
  enabled: true
  labeled_ratio: 0.1
```

### Python Configuration

```python
# Your custom augmentation configuration
augmentations = {
    'rotation': True,
    'rotation_degrees': 15,
    'flip': True,
    'flip_p': 0.5,
    'color_jitter': True,
    'brightness': 0.2,
    'contrast': 0.2,
}

# Implement your augmentation function based on config
def my_augmentations(image, mask):
    import random
    from torchvision.transforms.functional import hflip, rotate
    
    if augmentations['flip'] and random.random() < augmentations['flip_p']:
        image = hflip(image)
        mask = hflip(mask)
    
    if augmentations['rotation'] and random.random() < 0.3:
        angle = random.uniform(-augmentations['rotation_degrees'], augmentations['rotation_degrees'])
        image = rotate(image, angle)
        mask = rotate(mask, angle)
    
    return image, mask

# Create dataset
dataset = MySegmentationDataset(
    root_dir="./data/my_dataset",
    augmentations=my_augmentations
)
```

## Key Points

1. **User Responsibility**: Users implement their own datasets and augmentations
2. **Internal Augmentations**: Augmentation functions are internal to the datasets module
3. **Configuration References**: Configuration functions provide reference implementations
4. **Simplicity**: No complex base classes or abstractions
5. **Compatibility**: Works with any PyTorch dataset
6. **Semi-Supervised**: Built-in support for labeled/unlabeled data splits

## Dependencies

- **Core**: torch, torchvision, PIL, numpy
- **Optional**: albumentations, opencv-python (for advanced augmentations in your own implementations)

## Examples

See `examples/simple_usage_example.py` for comprehensive usage examples covering:

1. Basic dataset usage
2. Semi-supervised learning setup
3. Custom augmentations
4. Dataset statistics
5. Different augmentation strategies
6. Using configuration references
7. Complete workflow
