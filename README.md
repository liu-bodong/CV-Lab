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
   - `summary.yaml`: Summary of the training configuration.
   - `plot.png`: Visualization of training metrics (example as follows).
![example output](https://github.com/liu-bodong/CV-Lab/blob/main/runs/unet_0727_1906/plot.png)

4. There are jupyter notebooks in the `notebooks/` directory for additional functionalities:
   - `plot_csv.ipynb`: Generating plots from existing runs
   - `main.ipynb`: Vlidating model and visualizing train model's output and ground truch
   - `wandb.ipynb`: Testing wandb connection
   - `model_sanity`: Sanity check for models
   - `export`: Export trained `.pth` files to other formats (now only supports `.onnx`)
