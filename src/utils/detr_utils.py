"""
Utility functions for DETR-specific data formatting.
"""

from typing import List, Dict
import torch


def prepare_targets_for_detr(targets: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    """
    Adjust target labels for DETR.

    DETR in torchvision does NOT use a background class, so class indices
    must be in the range [0, num_classes - 1].
    Our KITTI dataset uses 1-based indexing (1 to 7). We shift them to 0-based (0 to 6).

    Args:
        targets: List of target dictionaries from the DataLoader.

    Returns:
        List of adjusted target dictionaries.
    """
    detr_targets = []
    for target in targets:
        new_target = {
            'boxes': target['boxes'],
            # Shift labels by -1 for DETR (e.g., 1->0, 7->6)
            'labels': target['labels'] - 1,
            'image_id': target['image_id']
        }
        detr_targets.append(new_target)

    return detr_targets