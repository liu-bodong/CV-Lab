# Datasets 

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os

__all__ = ['BrainMRIDataset']

class BrainMRIDataset(Dataset):
    def __init__(self, root_dir, image_size: tuple, transform=None, ):
        self.image_mask_pairs = []
        self.image_size = image_size
        
        if transform is None:
            # Image transforms (with normalization)
            self.image_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.image_transform = transform
        
        # Mask transforms (no normalization, just resize and convert to tensor)
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        
        print(f"Image transform: {self.image_transform}")
        print(f"Mask transform: {self.mask_transform}")
        

        # Loop through all patient folders
        for patient_folder in os.listdir(root_dir):
            patient_path = os.path.join(root_dir, patient_folder)
            if not os.path.isdir(patient_path):
                continue

            # Collect image–mask pairs
            for file in os.listdir(patient_path):
                if file.endswith(".tif") and "_mask" not in file:
                    image_path = os.path.join(patient_path, file)
                    mask_path = image_path.replace(".tif", "_mask.tif")
                    if os.path.exists(mask_path):
                        self.image_mask_pairs.append((image_path, mask_path))

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.image_mask_pairs[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Apply different transforms for image and mask
        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        mask = (mask > 0).float()  # Binary mask

        return image, mask