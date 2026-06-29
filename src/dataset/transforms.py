"""
Data augmentation and transformation functions using Albumentations.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size: int = 512) -> A.Compose:
    """Training transforms with bounding box support."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=25, min_visibility=0.3))

def get_val_transforms(img_size: int = 512) -> A.Compose:
    """Validation transforms (no augmentations)."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))