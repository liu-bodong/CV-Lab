# CV-Lab: Deep Learning for Image Segmentation

A comprehensive framework for training and evaluating deep learning models on medical image segmentation tasks, with support for multiple architectures and datasets.

## Overview

This repository provides an end-to-end solution for image/semantic segmentation tasks, particularly focused on medical imaging. It offers a clean, modular codebase that balances ease of use for beginners with the flexibility needed by experienced researchers.

### Key Highlights
- **Multiple Architecture Support**: U-Net and Attention U-Net implementations
- **Medical Image Focus**: Pre-configured for brain MRI, brain CT, and heart MRI datasets
- **Research-Ready**: Comprehensive logging, visualization, and experiment tracking
- **Production-Ready**: Model export capabilities (ONNX format)

## Features

- **Modular Architecture**: Well-structured codebase with reusable building blocks
- **Multiple Models**: U-Net and Attention U-Net with customizable depth and channels
- **Comprehensive Logging**: Training metrics, loss curves, and model checkpoints
- **Easy Configuration**: YAML-based hyperparameter management
- **Visualization Tools**: Built-in plotting and Jupyter notebook integration
- **Experiment Tracking**: Optional Weights & Biases (wandb) integration
- **Model Export**: Convert trained models to ONNX format for deployment

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
- CUDA (optional, but you definitely don't want to train on CPU)

## Quick Start

### 1. Configure Your Experiment
Edit the `hyper.yaml` file to set your hyperparameters:

```yaml
# Model configuration
model_type: attention_unet  # or 'unet'
image_size: [256, 256]
input_channels: 3
output_channels: 1
# more...

# Training configuration
batch_size: 16
epochs: 100
lr: 0.0003
# more...
```

### 2. Start Training
```bash
python train.py --config hyper.yaml
```

> **Note**: The `--config` parameter is optional. The training pipeline uses `hyper.yaml` by default. You can optionally specify a different configuration file.

*(Optional)*: Enable experiment tracking by configuring wandb in your YAML file:

```yaml
wandb_project: my_segmentation_project
wandb_entity: your_username
``` 

> **Note**: Currently, this repository does not contain any dataset data, you are expected to download dataset and write your own preprocessing code. We might add datasets and corresponding preprocessing in future.

### 3. Monitor Results
Training outputs are automatically saved to the `runs/` directory with the following structure:
```
runs/
└── [model_name]_[mmdd]_[HHMM]/
    ├── metrics.csv              # Training metrics per epoch
    ├── best.pth                 # Best model checkpoint
    ├── last.pth                 # Final model checkpoint
    ├── summary.yaml             # Experiment configuration and results
    └── plot.png                 # Training curves visualization
```

![Training Example](https://github.com/liu-bodong/CV-Lab/blob/main/runs/unet_0727_1906/plot.png)

## Available Models

### U-Net
Standard U-Net architecture for semantic segmentation with skip connections.

### Attention U-Net
Enhanced U-Net with attention gates for improved feature selection and localization.

### More networks and semi-supervised frameworks are under development...

## Jupyter Notebooks

The `notebooks/` directory provides specialized tools for different aspects of the workflow:

| Notebook | Purpose |
|----------|---------|
| **`main.ipynb`** | Model validation, output visualization, and ground truth comparison |
| **`plot_csv.ipynb`** | Generate custom plots from existing training metrics |
| **`export.ipynb`** | Convert trained PyTorch models to ONNX format |
| **`model_sanity.ipynb`** | Architecture validation and debugging |
| **`wandb.ipynb`** | Test and configure Weights & Biases integration |

> **Note**: Be aware that `main.ipynb` was used for many tasks, so there exists deprecated codes, though usually commented out.

## Configuration

All training parameters are managed through YAML configuration files. The main configuration options include:

### Model Parameters
- `model_type`: Choose between `unet` or `attention_unet`
- `image_size`: Input image dimensions `[height, width]`
- `input_channels`: Number of input channels (1 for grayscale, 3 for RGB)
- `output_channels`: Number of output classes
- `channels`: List of channel sizes for each encoder/decoder level

### Training Parameters
- `batch_size`: Training batch size
- `epochs`: Maximum number of training epochs
- `lr`: Learning rate
- `optimizer`: Optimizer type (Adam, SGD, etc.)
- `loss`: Loss function (BCEWithLogitsLoss, etc.)
- `patience`: Early stopping patience
- `use_amp`: Enable automatic mixed precision training


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
