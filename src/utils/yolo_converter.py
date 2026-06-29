"""
Converter for KITTI dataset to YOLO format (Ultralytics).
"""

import os
import yaml
from typing import Dict, List
from tqdm import tqdm
import shutil

from src.utils.utils import KITTI_CLASSES


def convert_kitti_to_yolo_format(
        kitti_root: str,
        output_dir: str,
        split: str = 'train'
) -> str:
    """
    Convert KITTI annotations to YOLO format.

    Args:
        kitti_root: Root directory of KITTI dataset (containing 'image_2', 'label_2').
        output_dir: Directory to save YOLO-formatted data.
        split: Dataset split ('train' or 'val').

    Returns:
        Path to the generated data.yaml file.
    """
    # Create directories
    img_dir = os.path.join(output_dir, 'images', split)
    lbl_dir = os.path.join(output_dir, 'labels', split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    # Load image IDs
    split_file = os.path.join(kitti_root, f'ImageSets/{split}.txt')
    if os.path.exists(split_file):
        with open(split_file, 'r') as f:
            image_ids = [line.strip() for line in f.readlines()]
    else:
        image_ids = [
            os.path.splitext(f)[0] for f in os.listdir(os.path.join(kitti_root, 'image_2'))
            if f.endswith('.png')
        ]

    # KITTI image dimensions (standard)
    IMG_WIDTH = 1242
    IMG_HEIGHT = 375

    print(f"Converting {len(image_ids)} images for {split} split...")

    for image_id in tqdm(image_ids, desc=f"Converting {split}"):
        img_path = os.path.join(kitti_root, 'image_2', f'{image_id}.png')
        lbl_path = os.path.join(kitti_root, 'label_2', f'{image_id}.txt')

        # Copy image (or symlink for disk space efficiency)
        dst_img = os.path.join(img_dir, f'{image_id}.png')
        if not os.path.exists(dst_img):
            shutil.copy(img_path, dst_img)

        # Convert labels
        dst_lbl = os.path.join(lbl_dir, f'{image_id}.txt')
        with open(lbl_path, 'r') as f_in, open(dst_lbl, 'w') as f_out:
            for line in f_in:
                parts = line.strip().split()
                class_name = parts[0]

                if class_name == 'DontCare':
                    continue

                if class_name not in KITTI_CLASSES or KITTI_CLASSES[class_name] == 0:
                    continue

                # Shift class index to 0-based for YOLO (1->0, 7->6)
                class_idx = KITTI_CLASSES[class_name] - 1

                # Bounding box: [xmin, ymin, xmax, ymax]
                xmin = float(parts[4])
                ymin = float(parts[5])
                xmax = float(parts[6])
                ymax = float(parts[7])

                # Convert to YOLO format: [x_center, y_center, width, height] (normalized)
                x_center = ((xmin + xmax) / 2.0) / IMG_WIDTH
                y_center = ((ymin + ymax) / 2.0) / IMG_HEIGHT
                width = (xmax - xmin) / IMG_WIDTH
                height = (ymax - ymin) / IMG_HEIGHT

                f_out.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    # Generate data.yaml
    data_yaml_path = os.path.join(output_dir, 'data.yaml')
    data_dict = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': [k for k, v in KITTI_CLASSES.items() if v != 0]
    }

    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_dict, f, sort_keys=False)

    print(f"Conversion complete. data.yaml saved at: {data_yaml_path}")
    return data_yaml_path