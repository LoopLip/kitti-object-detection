"""
Visualization utilities for object detection results and training metrics.
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.figure import Figure

from src.evaluation.metrics import calculate_iou


def draw_predictions(
        image: torch.Tensor,
        target: Dict[str, torch.Tensor],
        prediction: Dict[str, torch.Tensor],
        class_names: List[str],
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.3
) -> Image.Image:
    """
    Draw bounding boxes on image with ground truth and predictions.

    Args:
        image: Input image tensor [C, H, W].
        target: Ground truth dictionary with 'boxes' and 'labels'.
        prediction: Prediction dictionary with 'boxes', 'labels', 'scores'.
        class_names: List of class names.
        iou_threshold: IoU threshold for matching predictions to GT.
        confidence_threshold: Confidence threshold for filtering predictions.

    Returns:
        PIL Image with drawn bounding boxes.
    """
    # Convert tensor to PIL Image
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy().transpose(1, 2, 0)
        # Denormalize if needed (assuming ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = image_np * std + mean
        image_np = np.clip(image_np, 0, 1) * 255
        image_pil = Image.fromarray(image_np.astype(np.uint8))
    else:
        image_pil = image.copy()

    draw = ImageDraw.Draw(image_pil)

    # Try to load a font, fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except:
        font = ImageFont.load_default()

    # Draw Ground Truth (Green)
    gt_boxes = target['boxes'].cpu().numpy()
    gt_labels = target['labels'].cpu().numpy()

    for box, label in zip(gt_boxes, gt_labels):
        xmin, ymin, xmax, ymax = box
        label_name = class_names[int(label) - 1] if label > 0 else 'Unknown'

        # Draw rectangle
        draw.rectangle([xmin, ymin, xmax, ymax], outline='green', width=3)

        # Draw label
        text = f"GT: {label_name}"
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.rectangle([xmin, ymin - text_height - 4, xmin + text_width + 4, ymin], fill='green')
        draw.text((xmin + 2, ymin - text_height - 2), text, fill='white', font=font)

    # Draw Predictions (Red/Blue)
    pred_boxes = prediction['boxes'].cpu().numpy()
    pred_labels = prediction['labels'].cpu().numpy()
    pred_scores = prediction['scores'].cpu().numpy()

    # Filter by confidence
    mask = pred_scores >= confidence_threshold
    pred_boxes = pred_boxes[mask]
    pred_labels = pred_labels[mask]
    pred_scores = pred_scores[mask]

    # Match predictions to GT
    if len(pred_boxes) > 0 and len(gt_boxes) > 0:
        iou_matrix = calculate_iou(
            torch.from_numpy(pred_boxes),
            torch.from_numpy(gt_boxes)
        ).numpy()
    else:
        iou_matrix = np.zeros((len(pred_boxes), 1))

    for idx, (box, label, score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
        xmin, ymin, xmax, ymax = box
        label_name = class_names[int(label) - 1] if label > 0 else 'Unknown'

        # Determine color based on IoU
        if len(gt_boxes) > 0:
            max_iou = iou_matrix[idx].max()
            color = 'blue' if max_iou >= iou_threshold else 'red'
        else:
            color = 'red'  # False positive

        # Draw rectangle
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)

        # Draw label and score
        text = f"{label_name}: {score:.2f}"
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.rectangle([xmin, ymin - text_height - 4, xmin + text_width + 4, ymin], fill=color)
        draw.text((xmin + 2, ymin - text_height - 2), text, fill='white', font=font)

    return image_pil


def plot_training_curves(
        log_dirs: List[str],
        model_names: List[str],
        metrics: List[str] = ['train_loss', 'val_loss', 'mAP@0.5'],
        save_path: Optional[str] = None
) -> None:
    """
    Plot training curves for multiple models.

    Args:
        log_dirs: List of directories containing training logs (CSV format).
        model_names: List of model names for legend.
        metrics: List of metrics to plot.
        save_path: Optional path to save the figure.
    """
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 5)

    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 5))

    if num_metrics == 1:
        axes = [axes]

    colors = sns.color_palette("husl", len(model_names))

    for idx, (log_dir, model_name, color) in enumerate(zip(log_dirs, model_names, colors)):
        log_file = Path(log_dir) / 'training_log.csv'

        if not log_file.exists():
            print(f"Warning: Log file not found at {log_file}")
            continue

        # Read CSV log
        df = pd.read_csv(log_file)

        for ax, metric in zip(axes, metrics):
            if metric in df.columns:
                ax.plot(df['epoch'], df[metric], label=model_name, color=color, linewidth=2)
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
                ax.set_title(metric.replace('_', ' ').title(), fontsize=14, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_metrics_comparison(
        metrics_df: pd.DataFrame,
        metric_name: str = 'mAP@0.5',
        save_path: Optional[str] = None
) -> None:
    """
    Plot bar chart comparing a specific metric across models.

    Args:
        metrics_df: DataFrame with columns ['model', metric_name].
        metric_name: Name of the metric to plot.
        save_path: Optional path to save the figure.
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    # Sort by metric value
    df_sorted = metrics_df.sort_values(metric_name, ascending=True)

    # Create bar plot
    bars = plt.barh(df_sorted['model'], df_sorted[metric_name], color=sns.color_palette("viridis", len(df_sorted)))

    # Add value labels on bars
    for bar, value in zip(bars, df_sorted[metric_name]):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{value:.3f}', va='center', fontsize=11, fontweight='bold')

    plt.xlabel(metric_name.replace('_', ' ').title(), fontsize=12)
    plt.title(f'{metric_name.replace("_", " ").title()} Comparison', fontsize=14, fontweight='bold')
    plt.xlim(0, 1.0)
    plt.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()