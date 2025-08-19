"""
Logs for traininf of models, saves best model, plot, and csv file.
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

# Add src and networks to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from networks.unet import UNet
from networks.attention_unet import AttnUNet
from metrics import dice_loss, dice_coefficient

__all__ = ['log_results']

def log_results(config: dict, history: list, best_model_path: str, last_model_path: str):
    """
    Logs training results to a unique directory.

    Args:
        config (dict): Configuration dictionary.
        history (list): List of metrics per epoch.
        best_model_path (str): Path to the best model checkpoint.
        last_model_path (str): Path to the last model checkpoint.
    """
    if not history:
        print("No history to log. Exiting.")
        return

    # Create a unique directory for this run
    date = datetime.now().strftime("%m%d_%H%M")
    run_name = f"{config['model_type']}_{date}"
    output_dir = os.path.join(config.get('log_dir', 'runs'), run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Logging results to {output_dir}")

    # 1. Log model summary, param count, and FLOPs
    model_type = config['model_type']
    if model_type == "unet":
        model = UNet(in_channels=config['input_channels'], out_channels=config['output_channels'], channels=config['channels'])
    else:
        model = AttnUNet(in_channels=config['input_channels'], out_channels=config['output_channels'], channels=config['channels'])
    
    dummy_input = torch.randn(1, config['input_channels'], *tuple(config['image_size']))
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    # --- Plotting code moved to notebook for easier adjustment ---
    # (See notebook for interactive plotting)
    # Find best epoch
    best_epoch_metrics = max(history, key=lambda x: x['val_dice'])
    
    summary = config.copy()
    summary['param_count'] = f"{params/1e6:.2f}M"
    summary['flops'] = f"{flops/1e9:.2f}G"
    summary['best_epoch'] = best_epoch_metrics['epoch']
    summary['best_val_dice'] = f"{best_epoch_metrics['val_dice']:.4f}"
    summary['last_epoch'] = history[-1]['epoch']
    summary['last_val_dice'] = f"{history[-1]['val_dice']:.4f}"

    with open(os.path.join(output_dir, 'summary.yaml'), 'w') as f:
        yaml.dump(summary, f, sort_keys=False)

    # 2. Log metrics to CSV
    metrics_df = pd.DataFrame(history)
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)

    # 3. Plot and save metrics
    plt.style.use('petroff10')
    
    x   = metrics_df['epoch']
    y11 = metrics_df['train_loss']
    y12 = metrics_df['val_loss']
    y2  = metrics_df['val_dice']

    fig, ax1 = plt.subplots(figsize=(7, 4))

    # Plot Loss
    ax1.set_xlabel('Epoch', fontsize=12, loc='center')
    ax1.set_ylabel('Loss', color='brown', fontsize=12)
    ax1.plot(x, y11, '-', color='brown', label='Train Loss', zorder=22)
    ax1.plot(x, y12, '--', color='brown', label='Val Loss', zorder=12)
    ax1.tick_params(axis='y', labelcolor='brown')
    # ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune=None)) # Keep nbins consistent
    y1_top = math.ceil(max(y11.max(), y12.max()) * 10) / 10 # Add 10% margin
    ax1.set_ylim(bottom=0, top=y1_top)
    y1_ticks = np.linspace(0, y1_top, num=6)    
    y1_ticks = np.insert(y1_ticks, 1, y1_ticks[0] + (y1_ticks[1] - y1_ticks[0]) / 2) # Ensure 0 is included
    ax1.set_yticks(y1_ticks)

    # Set integer ticks for x-axis
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.set_xlim(left=x.min(), right=x.max())

    # Right y-axis for Dice
    ax2 = ax1.twinx()
    ax2.set_ylabel('Dice Score', color='darkblue', fontsize=12)
    ax2.plot(x, y2, '-', color='darkblue', label='Val Dice')
    ax2.tick_params(axis='y', labelcolor='darkblue')

    # ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10, prune=None)) # Keep nbins consistent
    ax2.set_ylim(bottom=0, top=1)
    y2_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
    ax2.set_yticks(y2_ticks)


    # Mark best epoch
    ax2.axvline(x=best_epoch_metrics['epoch'], color='seagreen', linestyle='--', linewidth=1.8, label=f"Best Epoch ({best_epoch_metrics['epoch']})")
    ax2.scatter(best_epoch_metrics['epoch'], best_epoch_metrics['val_dice'], color='sandybrown', marker='*', s=200, zorder=10, label=f"Best Dice: {best_epoch_metrics['val_dice']:.4f}")

    # Set spines color and linewidth
    ax2.spines['left'].set_color('brown')
    ax2.spines['right'].set_color('darkblue')

    ax2.spines['left'].set_linewidth(1.8)
    ax2.spines['right'].set_linewidth(1.8)

    plt.tick_params(direction="out")

    # Create a single legend on ax1 (or ax2, or the figure)
    fig.legend(loc='lower center', fontsize=9, frameon=False, ncol=5)
    plt.subplots_adjust(bottom=0.18)


    plt.title(f"Training Metrics: {config['model_type']} on {config['dataset']}", fontsize=14, pad=8)

    ax1.grid(True, linestyle='--', linewidth=1, alpha=0.5, zorder=0)
    ax2.grid(True, linestyle='--', linewidth=1, alpha=0.5, zorder=0)
    plt.title(f"Training Metrics: {config['model_type']} on {config['dataset']}",
              fontsize=14, pad=10)
    plt.savefig(os.path.join(output_dir, 'plot.png'))
    plt.close()

    # 4. Save the best model
    if best_model_path and os.path.exists(best_model_path):
        final_model_path = os.path.join(output_dir, os.path.basename(best_model_path))
        os.rename(best_model_path, final_model_path)
        print(f"Best model moved to {final_model_path}")
    else:
        print("Warning: Best model path not found.")
    
    if last_model_path and os.path.exists(last_model_path):
        final_last_model_path = os.path.join(output_dir, os.path.basename(last_model_path))
        os.rename(last_model_path, final_last_model_path)
        print(f"Last model moved to {final_last_model_path}")