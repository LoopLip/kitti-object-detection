import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn


def _get_efficientdet(num_classes: int, pretrained: bool) -> torch.nn.Module:
    """
    Initialize RetinaNet with ResNet50-FPN backbone (as EfficientDet proxy).
    Replaces the classification head to match the required number of classes.

    Args:
        num_classes: Number of classes (including background, e.g., 8).
        pretrained: Whether to use pretrained weights.

    Returns:
        RetinaNet model.
    """
    weights = torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = retinanet_resnet50_fpn(weights=weights)

    # Replace the classification head logits to match num_classes
    in_channels = model.backbone.out_channels
    num_anchors = model.head.classification_head.num_anchors

    model.head.classification_head.num_classes = num_classes
    model.head.classification_head.cls_logits = torch.nn.Conv2d(
        in_channels,
        num_anchors * num_classes,
        kernel_size=3,
        stride=1,
        padding=1
    )

    return model