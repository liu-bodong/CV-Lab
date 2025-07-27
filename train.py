"""
Main training script for U-Net segmentation models.
Organized for clarity, modularity, and robust logging.
"""
import argparse
import os
import sys
from matplotlib import ticker
import yaml
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from thop import profile
from tqdm import tqdm
import math
import numpy as np
import wandb

# Add src and networks to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from networks.unet import UNet
from networks.attention_unet import AttnUNet
import src.data_utils as data_utils
from src.metrics import dice_loss, dice_coefficient
from src.logger import log_results

run = None  # Global variable for wandb run

def train_model(config: dict, run):
    """
    Returns:
        tuple: A tuple containing:
            - list: A list of dictionaries with training/validation metrics per epoch.
            - str: The file path to the best performing model checkpoint.
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    # [TODO] Need to be modified in future for flexibility
    model_type = config['model_type']
    if model_type == "unet":
        model = UNet(
            in_channels=config['input_channels'],
            out_channels=config['output_channels'],
            channels=config['channels']
        )
    elif model_type == "attention_unet":
        model = AttnUNet(
            in_channels=config['input_channels'],
            out_channels=config['output_channels'],
            channels=config['channels']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    model.to(device)

    # Create data loaders
    train_loader, val_loader = data_utils.create_split_loaders(
        dataset=config['dataset'],
        root_dir=config['data_dir'],
        image_size=tuple(config['image_size']),
        batch_size=config['batch_size'],
        val_split=config['val_split']
    )

    # [TODO] Add support for different optimizers and loss functions
    # Optimizer, loss function, and AMP scaler
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = torch.nn.BCEWithLogitsLoss()
    scaler = None #torch.cuda.amp.GradScaler(enabled=config.get('use_amp', False))

    # Training loop
    history = []
    best_val_dice = 0.0
    best_model_path = None

    for epoch in range(config['epochs']):
        # --- Training Phase ---
        model.train()
        train_loss_epoch = 0.0
        
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [T]", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # [TODO] Add support for mixed precision training
            # with torch.cuda.amp.autocast(enabled=config.get('use_amp', False)):
            #     outputs = model(images)
            #     loss = criterion(outputs, masks)
            
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            
            
            train_loss_epoch += loss.item()

        avg_train_loss = train_loss_epoch / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss_epoch = 0.0
        val_dice_epoch = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [V]", leave=False):
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                dice = dice_coefficient(outputs, y)

                val_loss_epoch += loss.item()
                val_dice_epoch += dice.item()

        avg_val_loss = val_loss_epoch / len(val_loader)
        avg_val_dice = val_dice_epoch / len(val_loader)

        print(
            f"Epoch {epoch+1}/{config['epochs']} -> "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Dice: {avg_val_dice:.4f}"
        )

        # Log metrics
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_dice': avg_val_dice
        })
        
        run.log({
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_dice': avg_val_dice
        })

        threshold = config.get('min_improvement', 0.001)
        save_after_epochs = config.get('save_after_epochs', 1)
        # Save best model
        if epoch > save_after_epochs and avg_val_dice >= best_val_dice + threshold:
            best_val_dice = avg_val_dice
            os.makedirs(config['save_dir'], exist_ok=True)
            best_model_path = os.path.join(config['save_dir'], f"{config['model_type']}_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"-> New best model saved to {best_model_path} (Dice: {best_val_dice:.4f})")
        
        # End if converged
        if avg_train_loss - history[-1]['train_loss'] < 1e-5 and epoch > save_after_epochs:
            print(f"Convergence reached at epoch {epoch + 1}. Stopping training.")
            break

    return history, best_model_path


def main():
    """
    Main entry point for the training script.
    """
    parser = argparse.ArgumentParser(description="Train U-Net model.")
    parser.add_argument('--config', type=str, default='hyper.yaml', help='Path to config YAML file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("--- Training Configuration ---")
    print(yaml.dump(config, sort_keys=False))
    print("-----------------------------")

    run = wandb.init(
        project=config.get('wandb_project'),
        entity=config.get('wandb_entity'),
        config=config,
        name=f"{config['model_type']}_{datetime.now().strftime('%m%d_%H%M')}",
        mode="offline"
    )

    history, best_model_path = train_model(config, run)
    
    run.finish()
    
    if history:
        log_results(config, history, best_model_path)
        print("--- Training and Logging Completed ---")
    else:
        print("--- Training did not produce results to log ---")

if __name__ == "__main__":
    main()
