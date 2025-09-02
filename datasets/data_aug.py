"""
Essential augmentation utilities for image segmentation tasks.
Provides simple, reusable augmentation functions that can be easily applied to any dataset.
"""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image, ImageFilter
import random
from typing import Tuple, Optional, Union, List
import albumentations as A

__all__ = ["random_rotation", "random_flip", "random_crop", "color_jitter", "gaussian_blur", "elastic_transform", "grid_distortion", "random_brightness_contrast"]


def random_rotation(image, mask=None, degrees=15):
    """Apply random rotation to image and mask."""
    angle = random.uniform(-degrees, degrees)
    
    if isinstance(image, torch.Tensor):
        image = TF.rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR)
        if mask is not None:
            mask = TF.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)
    else:
        image = image.rotate(angle, resample=Image.BILINEAR)
        if mask is not None:
            mask = mask.rotate(angle, resample=Image.NEAREST)
    
    return image, mask


def random_flip(image, mask=None, horizontal=True, vertical=False):
    """Apply random horizontal and/or vertical flip."""
    if horizontal and random.random() < 0.5:
        if isinstance(image, torch.Tensor):
            image = TF.hflip(image)
            if mask is not None:
                mask = TF.hflip(mask)
        else:
            image = TF.hflip(image)
            if mask is not None:
                mask = TF.hflip(mask)
    
    if vertical and random.random() < 0.5:
        if isinstance(image, torch.Tensor):
            image = TF.vflip(image)
            if mask is not None:
                mask = TF.vflip(mask)
        else:
            image = TF.vflip(image)
            if mask is not None:
                mask = TF.vflip(mask)
    
    return image, mask


def random_crop(image, mask=None, size=(128, 128)):
    """Apply random crop to image and mask."""
    if isinstance(image, torch.Tensor):
        h, w = image.shape[-2:]
        th, tw = size
        
        if h < th or w < tw:
            # Pad if image is smaller than crop size
            pad_h = max(0, th - h)
            pad_w = max(0, tw - w)
            image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode='reflect')
            if mask is not None:
                mask = torch.nn.functional.pad(mask, (0, pad_w, 0, pad_h), mode='reflect')
            h, w = image.shape[-2:]
        
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        
        image = image[..., i:i+th, j:j+tw]
        if mask is not None:
            mask = mask[..., i:i+th, j:j+tw]
    else:
        # Handle PIL images
        image = TF.crop(image, *TF.get_random_crop_params(image, size))
        if mask is not None:
            mask = TF.crop(mask, *TF.get_random_crop_params(mask, size))
    
    return image, mask


def color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    """Apply color jitter to image."""
    if isinstance(image, torch.Tensor):
        image = TF.adjust_brightness(image, 1.0 + random.uniform(-brightness, brightness))
        image = TF.adjust_contrast(image, 1.0 + random.uniform(-contrast, contrast))
        image = TF.adjust_saturation(image, 1.0 + random.uniform(-saturation, saturation))
        image = TF.adjust_hue(image, random.uniform(-hue, hue))
    else:
        image = TF.adjust_brightness(image, 1.0 + random.uniform(-brightness, brightness))
        image = TF.adjust_contrast(image, 1.0 + random.uniform(-contrast, contrast))
        image = TF.adjust_saturation(image, 1.0 + random.uniform(-saturation, saturation))
        image = TF.adjust_hue(image, random.uniform(-hue, hue))
    
    return image


def gaussian_blur(image, radius_range=(0.1, 2.0)):
    """Apply Gaussian blur to image."""
    radius = random.uniform(*radius_range)
    
    if isinstance(image, torch.Tensor):
        image = TF.gaussian_blur(image, kernel_size=3, sigma=radius)
    else:
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    return image


def elastic_transform(image, mask=None, alpha=1.0, sigma=50.0, alpha_affine=50.0):
    """Apply elastic transform using albumentations."""
    # Convert to numpy for albumentations
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).numpy()
        mask_np = mask.permute(1, 2, 0).numpy() if mask is not None else None
    else:
        image_np = np.array(image)
        mask_np = np.array(mask) if mask is not None else None
    
    # Apply elastic transform
    transform = A.ElasticTransform(
        alpha=alpha,
        sigma=sigma,
        alpha_affine=alpha_affine,
        p=1.0
    )
    
    if mask_np is not None:
        transformed = transform(image=image_np, mask=mask_np)
        image_np = transformed['image']
        mask_np = transformed['mask']
    else:
        transformed = transform(image=image_np)
        image_np = transformed['image']
    
    # Convert back to original format
    if isinstance(image, torch.Tensor):
        image = torch.from_numpy(image_np).permute(2, 0, 1)
        if mask is not None:
            mask = torch.from_numpy(mask_np).permute(2, 0, 1)
    else:
        image = Image.fromarray(image_np)
        if mask is not None:
            mask = Image.fromarray(mask_np)
    
    return image, mask


def grid_distortion(image, mask=None, num_steps=5, distort_limit=0.3):
    """Apply grid distortion using albumentations."""
    # Convert to numpy for albumentations
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).numpy()
        mask_np = mask.permute(1, 2, 0).numpy() if mask is not None else None
    else:
        image_np = np.array(image)
        mask_np = np.array(mask) if mask is not None else None
    
    # Apply grid distortion
    transform = A.GridDistortion(
        num_steps=num_steps,
        distort_limit=distort_limit,
        p=1.0
    )
    
    if mask_np is not None:
        transformed = transform(image=image_np, mask=mask_np)
        image_np = transformed['image']
        mask_np = transformed['mask']
    else:
        transformed = transform(image=image_np)
        image_np = transformed['image']
    
    # Convert back to original format
    if isinstance(image, torch.Tensor):
        image = torch.from_numpy(image_np).permute(2, 0, 1)
        if mask is not None:
            mask = torch.from_numpy(mask_np).permute(2, 0, 1)
    else:
        image = Image.fromarray(image_np)
        if mask is not None:
            mask = Image.fromarray(mask_np)
    
    return image, mask


def random_brightness_contrast(image, brightness_limit=0.2, contrast_limit=0.2):
    """Apply random brightness and contrast adjustment."""
    # Convert to numpy for albumentations
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).numpy()
    else:
        image_np = np.array(image)
    
    # Apply brightness/contrast adjustment
    transform = A.RandomBrightnessContrast(
        brightness_limit=brightness_limit,
        contrast_limit=contrast_limit,
        p=1.0
    )
    
    transformed = transform(image=image_np)
    image_np = transformed['image']
    
    # Convert back to original format
    if isinstance(image, torch.Tensor):
        image = torch.from_numpy(image_np).permute(2, 0, 1)
    else:
        image = Image.fromarray(image_np)
    
    return image


def random_gamma(image, gamma_limit=(80, 120)):
    """Apply random gamma correction."""
    gamma = random.uniform(*gamma_limit) / 100.0
    
    if isinstance(image, torch.Tensor):
        image = TF.adjust_gamma(image, gamma)
    else:
        image = TF.adjust_gamma(image, gamma)
    
    return image
