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

# Add src and networks to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from networks.unet import UNet
from networks.attention_unet import AttnUNet
import src.data_utils as data_utils
from src.metrics import dice_loss, dice_coefficient

def train_model(config: dict):
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

def log_results(config: dict, history: list, best_model_path: str):
    """
    Logs training results to a unique directory.

    Args:
        config (dict): Configuration dictionary.
        history (list): List of metrics per epoch.
        best_model_path (str): Path to the best model checkpoint.
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

    fig, ax1 = plt.subplots(figsize=(6, 4))

    # Plot Loss
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='brown', fontsize=10)
    ax1.plot(x, y11, '-', color='brown', label='Train Loss')
    ax1.plot(x, y12, '--', color='brown', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor='brown')
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8, prune=None)) # Keep nbins consistent
    ax1.set_ylim(bottom=0)

    # Set integer ticks for x-axis
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.set_xlim(left=x.min(), right=x.max())

    # Right y-axis for Dice
    ax2 = ax1.twinx()
    ax2.set_ylabel('Dice Coefficient', color='darkblue', fontsize=10)
    ax2.plot(x, y2, '-', color='darkblue', label='Val Dice')
    ax2.tick_params(axis='y', labelcolor='darkblue')

    ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8, prune=None)) # Keep nbins consistent
    ax2.set_ylim(bottom=0, top=1)


    # Mark best epoch
    ax2.axvline(x=best_epoch_metrics['epoch'], color='seagreen', linestyle=':', linewidth=2, label=f"Best Epoch ({best_epoch_metrics['epoch']})")
    ax2.scatter(best_epoch_metrics['epoch'], best_epoch_metrics['val_dice'], color='sandybrown', marker='*', s=230, zorder=5, label=f"Best Dice: {best_epoch_metrics['val_dice']:.4f}")


    # --- Consolidate Legends ---
    # Get handles and labels from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # Combine them
    lines = lines1 + lines2
    labels = labels1 + labels2

    # Set spines color and linewidth
    ax2.spines['left'].set_color('brown')
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['right'].set_color('darkblue')
    ax2.spines['right'].set_linewidth(1.5)

    plt.tick_params(direction="out")

    # Create a single legend on ax1 (or ax2, or the figure)
    legend = ax1.legend(lines, labels, fontsize=9, loc='upper left', bbox_to_anchor=(0.08, 1), frameon=True, fancybox=False, framealpha=0) # Adjust bbox_to_anchor if needed

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

    history, best_model_path = train_model(config)
    
    if history:
        log_results(config, history, best_model_path)
        print("--- Training and Logging Completed ---")
    else:
        print("--- Training did not produce results to log ---")

if __name__ == "__main__":
    main()
