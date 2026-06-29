"""
Model factory and initialization for all object detection architectures.
"""

from typing import Union
import torch
import torch.nn as nn

from src.models.faster_rcnn import _get_faster_rcnn
from src.models.ssd import _get_ssd
from src.models.efficientdet import _get_efficientdet
from src.models.detr import _get_detr
from src.models.yolo import YOLOWrapper


def get_model(
        model_name: str,
        num_classes: int,
        pretrained: bool = True
) -> Union[nn.Module, YOLOWrapper]:
    """
    Factory function to initialize object detection models.

    Args:
        model_name: Name of the model architecture.
                    Options: 'faster_rcnn', 'ssd', 'efficientdet', 'detr', 'yolo'.
        num_classes: Number of output classes.
                     NOTE: For DETR, pass the number of foreground classes (e.g., 7).
                     For others, include the background class (e.g., 8).
        pretrained: Whether to load pretrained weights.

    Returns:
        Initialized PyTorch model or YOLOWrapper.

    Raises:
        ValueError: If model_name is not supported.
    """
    model_name = model_name.lower()

    if model_name == 'faster_rcnn':
        return _get_faster_rcnn(num_classes, pretrained)
    elif model_name == 'ssd':
        return _get_ssd(num_classes, pretrained)
    elif model_name == 'efficientdet':
        return _get_efficientdet(num_classes, pretrained)
    elif model_name == 'detr':
        return _get_detr(num_classes, pretrained)
    elif model_name == 'yolo':
        # For YOLO, num_classes is handled via data.yaml, but we pass it for consistency
        return YOLOWrapper(model_name='yolov8n.pt', num_classes=num_classes)
    else:
        raise ValueError(
            f"Model '{model_name}' is not supported. "
            "Choose from: 'faster_rcnn', 'ssd', 'efficientdet', 'detr', 'yolo'."
        )