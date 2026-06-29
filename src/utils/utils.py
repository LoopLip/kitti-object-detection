"""
Utility functions and constants for KITTI dataset processing.
"""

from typing import Dict, List, Tuple
import torch

# Class mapping for KITTI 2D object detection
# 'DontCare' is assigned index 0 and will be filtered out during training
KITTI_CLASSES: Dict[str, int] = {
    'Car': 1,
    'Van': 2,
    'Truck': 3,
    'Pedestrian': 4,
    'Cyclist': 5,
    'Tram': 6,
    'Person_sitting': 7,
    'DontCare': 0
}

NUM_CLASSES: int = len(KITTI_CLASSES) - 1  # Exclude 'DontCare'


def parse_kitti_annotation(annotation_line: str) -> Tuple[str, List[float]]:
    """
    Parse a single line from KITTI annotation file.

    Args:
        annotation_line: String containing annotation data.

    Returns:
        Tuple of (class_name, bounding_box_coordinates).
        bounding_box_coordinates: [xmin, ymin, xmax, ymax]
    """
    parts = annotation_line.strip().split()
    class_name = parts[0]

    # bbox format: [left, top, right, bottom]
    bbox = [float(x) for x in parts[4:8]]

    return class_name, bbox


def load_annotations(annotation_file: str) -> List[Dict]:
    """
    Load all annotations from a KITTI label file.

    Args:
        annotation_file: Path to the .txt annotation file.

    Returns:
        List of dictionaries containing 'class_name' and 'bbox'.
    """
    annotations = []

    with open(annotation_file, 'r') as f:
        for line in f:
            class_name, bbox = parse_kitti_annotation(line)

            # Filter out 'DontCare' class
            if class_name == 'DontCare':
                continue

            # Filter out very small objects (area < 25 pixels)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if width * height < 25:
                continue

            annotations.append({
                'class_name': class_name,
                'bbox': bbox
            })

    return annotations


def collate_fn(batch: List[Tuple]) -> Tuple:
    """
    Custom collate function for DataLoader to handle variable number of objects.

    Args:
        batch: List of tuples (image, target).

    Returns:
        Tuple of (images, targets).
    """
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    return torch.stack(images, dim=0), targets