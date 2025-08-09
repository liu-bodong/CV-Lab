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
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5).float()
    
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true) - intersection

    iou_score = (intersection + smooth) / (union + smooth)
    return iou_score


def sensitivity_recall(y_pred: torch.Tensor, y_true: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Calculates sensitivity (recall) for segmentation.
    Sensitivity = TP / (TP + FN)

    Args:
        y_pred (torch.Tensor): Predicted masks.
        y_true (torch.Tensor): Ground truth masks.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor: The sensitivity score.
    """
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5).float()
    
    true_positives = torch.sum(y_pred * y_true)
    false_negatives = torch.sum(y_true * (1 - y_pred))
    
    sensitivity = (true_positives + smooth) / (true_positives + false_negatives + smooth)
    return sensitivity


def specificity(y_pred: torch.Tensor, y_true: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Calculates specificity for segmentation.
    Specificity = TN / (TN + FP)

    Args:
        y_pred (torch.Tensor): Predicted masks.
        y_true (torch.Tensor): Ground truth masks.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor: The specificity score.
    """
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5).float()
    
    true_negatives = torch.sum((1 - y_pred) * (1 - y_true))
    false_positives = torch.sum(y_pred * (1 - y_true))
    
    spec = (true_negatives + smooth) / (true_negatives + false_positives + smooth)
    return spec


def precision(y_pred: torch.Tensor, y_true: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Calculates precision for segmentation.
    Precision = TP / (TP + FP)

    Args:
        y_pred (torch.Tensor): Predicted masks.
        y_true (torch.Tensor): Ground truth masks.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor: The precision score.
    """
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5).float()
    
    true_positives = torch.sum(y_pred * y_true)
    false_positives = torch.sum(y_pred * (1 - y_true))
    
    prec = (true_positives + smooth) / (true_positives + false_positives + smooth)
    return prec


def f1_score(y_pred: torch.Tensor, y_true: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Calculates F1 score for segmentation.
    F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_pred (torch.Tensor): Predicted masks.
        y_true (torch.Tensor): Ground truth masks.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor: The F1 score.
    """
    prec = precision(y_pred, y_true, smooth)
    recall = sensitivity_recall(y_pred, y_true, smooth)
    
    f1 = (2 * prec * recall + smooth) / (prec + recall + smooth)
    return f1


def compute_all_metrics(y_pred: torch.Tensor, y_true: torch.Tensor, smooth: float = 1.0) -> dict:
    """
    Computes all available metrics at once for efficiency.

    Args:
        y_pred (torch.Tensor): Predicted masks.
        y_true (torch.Tensor): Ground truth masks.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        dict: Dictionary containing all computed metrics.
    """
    return {
        'dice': dice_coefficient(y_pred, y_true, smooth),
        'iou': iou(y_pred, y_true, smooth),
        'sensitivity': sensitivity_recall(y_pred, y_true, smooth),
        'specificity': specificity(y_pred, y_true, smooth),
        'precision': precision(y_pred, y_true, smooth),
        'f1': f1_score(y_pred, y_true, smooth)
    }