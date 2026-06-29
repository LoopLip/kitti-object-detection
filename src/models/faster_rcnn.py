"""
Model factory and specific architectures for object detection.
"""

from typing import Dict, Any
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(model_name: str, num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    """
    Factory function to initialize object detection models.

    Args:
        model_name: Name of the model architecture ('faster_rcnn', 'yolo', etc.).
        num_classes: Number of output classes (including background).
        pretrained: Whether to load pretrained weights.

    Returns:
        Initialized PyTorch model.

    Raises:
        ValueError: If model_name is not supported.
    """
    if model_name == 'faster_rcnn':
        return _get_faster_rcnn(num_classes, pretrained)
    # elif model_name == 'yolo': ... (To be implemented in Stage 6)
    else:
        raise ValueError(f"Model '{model_name}' is not supported yet.")


def _get_faster_rcnn(num_classes: int, pretrained: bool) -> torch.nn.Module:
    """
    Initialize Faster R-CNN with ResNet50-FPN backbone.
    Replaces the box predictor head to match the required number of classes.

    Args:
        num_classes: Number of classes (e.g., 7 KITTI classes + 1 background = 8).
        pretrained: Whether to use ImageNet pretrained weights.

    Returns:
        Faster R-CNN model.
    """
    # Load pre-trained model (using modern torchvision API)
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one matching our num_classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model