"""
KITTI Dataset implementation for 2D object detection.
"""

import os
from typing import Callable, Dict, List, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.utils.utils import KITTI_CLASSES, load_annotations
from src.dataset.transforms import get_train_transforms, get_val_transforms


class KITTIDataset(Dataset):
    """
    PyTorch Dataset for KITTI 2D Object Detection.

    Attributes:
        root_dir: Root directory containing 'image_2' and 'label_2' folders.
        split: Dataset split ('train' or 'val').
        transforms: Optional transforms to apply to images.
        image_ids: List of image IDs (filenames without extension).
    """

    def __init__(
            self,
            root_dir: str,
            split: str = 'train',
            transforms: Optional[Callable] = None
    ) -> None:
        """
        Initialize KITTIDataset.

        Args:
            root_dir: Root directory of KITTI dataset.
            split: Dataset split ('train' or 'val').
            transforms: Optional transforms to apply.
        """
        self.root_dir = root_dir
        self.split = split

        # Set default transforms if not provided
        if transforms is None:
            if split == 'train':
                self.transforms = get_train_transforms()
            else:
                self.transforms = get_val_transforms()
        else:
            self.transforms = transforms

        # Load image IDs from split file
        split_file = os.path.join(root_dir, f'ImageSets/{split}.txt')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                self.image_ids = [line.strip() for line in f.readlines()]
        else:
            # Fallback: load all images from image_2 directory
            image_dir = os.path.join(root_dir, 'image_2')
            self.image_ids = [
                os.path.splitext(f)[0] for f in os.listdir(image_dir)
                if f.endswith('.png')
            ]

        print(f"Loaded {len(self.image_ids)} images for {split} split")

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            Tuple of (image, target).
            image: Tensor of shape [C, H, W].
            target: Dictionary with keys:
                - 'boxes': FloatTensor of shape [N, 4] in format [xmin, ymin, xmax, ymax]
                - 'labels': LongTensor of shape [N]
                - 'image_id': LongTensor of shape [1]
        """
        image_id = self.image_ids[idx]

        # Load image
        image_path = os.path.join(self.root_dir, 'image_2', f'{image_id}.png')
        image = Image.open(image_path).convert('RGB')

        # Load annotations
        annotation_path = os.path.join(self.root_dir, 'label_2', f'{image_id}.txt')
        annotations = load_annotations(annotation_path)

        # Convert annotations to tensors
        boxes = []
        labels = []

        for ann in annotations:
            class_name = ann['class_name']
            bbox = ann['bbox']

            # Skip if class not in mapping
            if class_name not in KITTI_CLASSES:
                continue

            label = KITTI_CLASSES[class_name]
            labels.append(label)
            boxes.append(bbox)

        # Handle case with no annotations
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        # Create target dictionary (COCO format for torchvision)
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx], dtype=torch.int64)
        }

        # Apply transforms
        if self.transforms is not None:
            # Albumentations expects numpy array and list of boxes
            import numpy as np
            image_np = np.array(image)

            # Convert boxes to list of lists for albumentations
            boxes_list = boxes.tolist() if len(boxes) > 0 else []
            labels_list = labels.tolist() if len(labels) > 0 else []

            transformed = self.transforms(image=image_np, bboxes=boxes_list, labels=labels_list)

            image = transformed['image']

            # Convert back to tensors
            if len(transformed['bboxes']) > 0:
                boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
                labels = torch.as_tensor(transformed['labels'], dtype=torch.int64)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)

            target['boxes'] = boxes
            target['labels'] = labels

        # Гарантируем, что image является тензором (убирает предупреждение IDE)
        if not isinstance(image, torch.Tensor):
            import torchvision.transforms.functional
            image = torchvision.transforms.functional.to_tensor(image)

        return image, target