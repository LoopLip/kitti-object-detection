"""
SSD (Single Shot MultiBox Detector) implementation using torchvision.
"""

import torch
import torchvision
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDHead


def _get_ssd(num_classes: int, pretrained: bool) -> torch.nn.Module:
    """
    Initialize SSD300 with VGG16 backbone.
    Replaces the SSDHead to match the required number of classes.

    Args:
        num_classes: Number of classes (including background, e.g., 8).
        pretrained: Whether to use pretrained weights.

    Returns:
        SSD model.
    """
    weights = torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT if pretrained else None
    model = ssd300_vgg16(weights=weights)

    # SSD300 specific architecture parameters
    in_channels = [512, 1024, 512, 256, 256, 256]
    num_anchors = [4, 6, 6, 6, 4, 4]

    # Replace the pre-trained head with a new one matching our num_classes
    model.head = SSDHead(in_channels, num_anchors, num_classes)

    return model