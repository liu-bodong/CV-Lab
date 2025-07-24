# scripts/train.py
import argparse
from pathlib import Path
import torch

from models.unet.attention_unet import AttnUNet
from training.trainer import SegmentationTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize model, data, trainer
    model = AttnUNet(config.input_channels, config.output_channels)
    trainer = SegmentationTrainer(model, config)
    
    # Train
    trainer.train()

if __name__ == "__main__":
    main()