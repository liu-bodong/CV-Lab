"""
Utility functions for calculating metrics for model evaluation and training.
"""
import torch


def dice_coefficient(y_pred: torch.Tensor, y_true: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Calculates the Dice coefficient for a batch of predictions.

    Args:
        y_pred (torch.Tensor): Predicted masks (batch_size, 1, H, W).
        y_true (torch.Tensor): Ground truth masks (batch_size, 1, H, W).
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor: The Dice coefficient.
    """
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5).float()

    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def dice_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Dice loss.

    Args:
        y_pred (torch.Tensor): Predicted masks.
        y_true (torch.Tensor): Ground truth masks.

    Returns:
        torch.Tensor: The Dice loss.
    """
    return 1 - dice_coefficient(y_pred, y_true)


def iou(y_pred: torch.Tensor, y_true: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Calculates the Intersection over Union (IoU) for a batch of predictions.

    Args:
        y_pred (torch.Tensor): Predicted masks (batch_size, 1, H, W).
        y_true (torch.Tensor): Ground truth masks (batch_size, 1, H, W).
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor: The IoU.
    """
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou