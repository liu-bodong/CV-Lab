# U-Nets
This repository stores U-Net like segmentation frameworks.

## Requirements
- Python 3.8+


## Repo Structure
U-Nets
├── README.md
├── requirements.txt
├── config/
│   ├── __init__.py
│   ├── base_config.py
│   ├── unet_config.py
│   └── attention_unet_config.py
├── data/
│   ├── raw/
│   │   ├── kaggle_3m/
│   │   ├── brats2020/
│   │   └── medical_decathlon/
│   ├── processed/
│   └── README.md
├── datasets/
│   ├── __init__.py
│   ├── base_dataset.py
│   ├── brain_mri_dataset.py
│   ├── medical_segmentation_dataset.py
│   └── transforms.py
├── models/
│   ├── __init__.py
│   ├── base_model.py
│   ├── blocks.py
│   ├── unet/
│   │   ├── __init__.py
│   │   ├── vanilla_unet.py
│   │   ├── attention_unet.py
│   │   ├── unet_plus_plus.py
│   │   └── residual_unet.py
│   ├── segnet/
│   └── deeplab/
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   ├── losses.py
│   ├── metrics.py
│   └── callbacks.py
├── utils/
│   ├── __init__.py
│   ├── visualization.py
│   ├── logging.py
│   └── checkpoints.py
├── experiments/
│   ├── unet_brain_mri/
│   │   ├── config.yaml
│   │   ├── train.py
│   │   └── results/
│   └── attention_unet_brain_mri/
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   ├── model_comparison.ipynb
│   └── visualization.ipynb
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── preprocess_data.py
└── saved_models/
    ├── unet/
    ├── attention_unet/
    └── checkpoints/