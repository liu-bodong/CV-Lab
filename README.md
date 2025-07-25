# Image Segmentation with U-Nets
This repository contains code and resources for training U-Net models for image segmentation tasks.

## Requirements
- Python 3.8+
- PyTorch 1.7+
- torchvision


## Usage
1. Edit the `hyper.yaml` file to set your hyperparameters.
2. Run the training script:
   ```bash
   python train.py --config hyper.yaml
   ```
3. The training logs and model checkpoints will be saved in the `runs/` directory. Each run will have its own subdirectory named with the model name and timestamp. Inside each run directory, you will find:
   - `metrics.csv`: Metrics logged during training.
   - `model.pth`: The trained model checkpoint.
   - 'summary.yaml': Summary of the training configuration.
   - 'plot.png': Visualization of training metrics.