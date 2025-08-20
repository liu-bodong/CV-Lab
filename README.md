# CV-Lab: Deep Learning for Image Segmentation

A modular framework for training and evaluating deep learning models on image segmentation tasks. While it does not offer fine-grained solutions out of the box, it is *ideal* for learning, rapid prototyping, and validating research ideas. The codebase is well-documented and straightforward to support both learning and easy extension.

## Overview

This repository provides tools for image segmentation using deep learning. It features a clean, modular codebase with support for common segmentation architectures and standard training workflows.

### Features
- **Multiple Architectures**: U-Net and Attention U-Net implementations
- **Modular Design**: Organized code structure with separate modules for datasets, networks, and utilities
- **Dynamic Configuration**: Reflection mechanism for automatic configuration loading and scalability
- **Configuration Management**: YAML-based hyperparameter configuration
- **Training Pipeline**: Standard training loop with validation and checkpointing
- **Experiment Tracking**: Optional Weights & Biases integration
- **Model Export**: ONNX format support for deployment

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/liu-bodong/CV-Lab.git
   cd CV-Lab
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Requirements
- Python 3.8+
- PyTorch 1.9.0+
- torchvision 0.10.0+
- CUDA

## Project Structure
Each sub-directory functions as a module with specified export components to be called using reflection.

```
CV-Lab/
├── datasets/         # Dataset classes and data loading utilities
├── networks/         # Model architectures (U-Net, Attention U-Net)
├── utils/            # Training utilities (logging, metrics, loss functions)
├── notebooks/        # Jupyter notebooks for analysis and visualization
├── runs/             # Training outputs and model checkpoints
├── train.py          # Main training script
└── hyper.yaml        # Configuration file
```

## Quick Start

### 1. Prepare Your Dataset
Implement your dataset class in `datasets/` or use the provided datasets as a reference. The framework expects dataset classes to return image-mask pairs.

### 2. Configure Training
Edit `hyper.yaml` to set your parameters. Note that the framework uses a dynamic configuration system implemented by reflection in python. Each sub-module exports classes and methods to be used. Please pay attention to the exact naming of imported components. Refer to the following example for a glimpse of hyper parameter editing:

```yaml
# Model configuration
model_type: ModelClass  
image_size: [256, 256]
input_channels: 3
output_channels: 1
channels: [64, 128, 256, 512] # for frameworks that require specific channel sizes

# Training configuration
batch_size: 16
epochs: 100
init_lr: 0.0003
dataset: DatasetClass
data_dir: ./data/your_dataset
```

### 3. Start Training
```bash
python train.py --config hyper.yaml
```

### 4. Monitor Results
Training outputs are saved to the `runs/` directory:
```
runs/[model_name]_[timestamp]/
├── metrics.csv      # Training metrics
├── best.pth         # Best model checkpoint
├── last.pth         # Final checkpoint
├── summary.yaml     # Run configuration
└── plot.png         # Training curves
```
An example of plot.png:
![Training Example](https://github.com/liu-bodong/CV-Lab/blob/main/runs/unet_0727_1906/plot.png)

## Available Models

Currently, the framework supports the following models:
- ResNet50/101
- MobileNetV1
- U-Net
- Attention U-Net
- more models are under development, feel free to add your own too!

## Datasets

The framework includes a `BrainMRIDataset` class corresponding to a [dataset on kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) as an example implementation. You can create custom dataset classes for your needs.

### BrainMRIDataset
- Loads image-mask pairs from organized directory structure
- Applies separate transforms for images and masks
- Supports common medical image formats

## Configuration Options

### Model Parameters
- `model_type`: Model architecture
- `image_size`: Input dimensions `[height, width]`
- `input_channels`: Number of input channels
- `output_channels`: Number of output classes
- `channels`: A list of channel sizes -- channel progression for encoder/decoder

### Training Parameters
- `batch_size`: Training batch size
- `epochs`: Number of training epochs
- `init_lr`: Initial learning rate
- `dataset`: Dataset class name
- `data_dir`: Path to dataset
- `val_split`: Validation split ratio

### Optional Features
- **Weights & Biases**: Set `wandb_project`, `wandb_entity`, and `wandb_mode` for experiment tracking
- **Early Stopping**: Configure `patience` for convergence early stopping
- **Learning Rate Scheduling**: Use `lr_rampdown_epochs` for learning rate decay


## Jupyter Notebooks

The `notebooks/` directory contains tools for analysis and visualization:

- **`main.ipynb`**: Model validation and output visualization
- **`plot_csv.ipynb`**: Generate plots from training metrics
- **`export.ipynb`**: Convert models to ONNX format
- **`model_sanity.ipynb`**: Architecture testing and debugging
- **`wandb.ipynb`**: Weights & Biases integration testing

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
