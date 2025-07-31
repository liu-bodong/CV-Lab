# Image segmentation with deep learning
Code and resources for training deep-learning models for image/semantic segmentation tasks. 

## Overview
This repo provides a comprehensive coding and experimenting infrastructure for training, testing, logging, and analyzing deep learning models and their runs. For beginners, it is easy to use; for experienced machine learning scientists, it is versatile and very customizable for additional needs.

## Features
- Popular deep learning blocks and networks for computer vision tasks
- Training and testing pipeline that is easy to use
- Centralized and easy-to-use hyper-parameter tuning with `.yaml` files
- Logging system that tracks training information and draws plots for learning, experimental, and replication purposes
- Lightweight jupyter notebooks for quick testing, debugging, and sanity checking
- (Optional) Incorporation of `wandb` package for logging, online monitoring, and more comprehensive analysis

## Requirements
- Python 3.8+
- PyTorch 1.7+
- torchvision

## Usage
1. Edit the `hyper.yaml` file to set your hyper-parameters.
2. Run the training script:
   ```bash
   python train.py --config [hyper-parameter file name].yaml
   ```
   The training pipeline by defualt uses the configuration from `hyper.yaml`. Therefore, he argument in the above command is not necessary, unless you want to use files other than `hyper.yaml` for testing purposes.
3. The training logs and model checkpoints will be saved in the `runs/` directory. Each run will have its own subdirectory named with the model name and timestamp in the format `name_yymm__HHMM/`. Inside each run directory, you will find:
   - `metrics.csv`: Metrics logged during training (for each epoch by default)
   - `[model_name]_best.pth`: The trained model checkpoint at best epoch
   - `[model_name]_last.pth`: The trained model checkpoint at last epoch
   - `summary.yaml`: Summary of the run, including hyper-parameters used and other training info
   - `plot.png`: Visualization of training metrics (example as follows)
![example output](https://github.com/liu-bodong/CV-Lab/blob/main/runs/unet_0727_1906/plot.png)

Additionally, you may use `wandb` package and platform for online monitoring and hyper-parameter fine-tuning. The relevant code can be easily found in `train.py`, you can modify it as how you like it to work. 

4. The `src/` directory contains useful utilities.
   - `data_utils.py`: Utilities for loading data
   - `logger.py`: Logger system for local logging
   - `metrics.py`: Metrics calculation

5. There are jupyter notebooks in the `notebooks/` directory for additional functionalities:
   - `plot_csv.ipynb`: Generating plots from existing runs's `metrics.csv` file
   - `main.ipynb`: Vlidating model and visualizing train model's output and ground truth. This is my old notebook for running everything so it contains deprecated codes, though those are usually commented out so do not worry too much
   - `wandb.ipynb`: Testing wandb connection
   - `model_sanity.ipynb`: Sanity check for models
   - `export.ipynb`: Export trained `.pth` files to other formats (now only supports `.onnx`)
   - `playground.ipynb`: A playground for anything
