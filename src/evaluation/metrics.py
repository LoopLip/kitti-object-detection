"""
Evaluation metrics for object detection: mAP, Precision, Recall, F1.
Implements COCO-style evaluation with vectorized IoU computation.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.detr_utils import prepare_targets_for_detr

def calculate_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU) between two sets of boxes.
    Fully vectorized implementation using PyTorch.

    Args:
        boxes1: Tensor of shape [N, 4] in format [xmin, ymin, xmax, ymax].
        boxes2: Tensor of shape [M, 4] in format [xmin, ymin, xmax, ymax].

    Returns:
        IoU matrix of shape [N, M].
    """
    # Area of boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]

    # Intersection coordinates
    inter_xmin = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])  # [N, M]
    inter_ymin = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])  # [N, M]
    inter_xmax = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])  # [N, M]
    inter_ymax = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])  # [N, M]

    # Intersection area
    inter_width = torch.clamp(inter_xmax - inter_xmin, min=0)
    inter_height = torch.clamp(inter_ymax - inter_ymin, min=0)
    inter_area = inter_width * inter_height  # [N, M]

    # Union area
    union_area = area1[:, None] + area2[None, :] - inter_area  # [N, M]

    # IoU
    iou = inter_area / torch.clamp(union_area, min=1e-6)

    return iou

def compute_precision_recall(
        predictions: List[Dict],
        ground_truths: List[Dict],
        class_id: int,
        iou_threshold: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Precision and Recall for a single class across all images.

    Args:
        predictions: List of prediction dicts per image.
        ground_truths: List of ground truth dicts per image.
        class_id: Class ID to evaluate.
        iou_threshold: IoU threshold for matching.

    Returns:
        Tuple of (precision, recall, confidence_scores) tensors.
    """
    # Collect all predictions for this class
    all_preds = []
    for img_idx, pred in enumerate(predictions):
        boxes = pred['boxes']
        labels = pred['labels']
        scores = pred['scores']

        mask = labels == class_id
        if mask.any():
            for i in range(mask.sum().item()):
                all_preds.append({
                    'image_id': img_idx,
                    'box': boxes[mask][i],
                    'score': scores[mask][i].item()
                })

    # Sort predictions by confidence score (descending)
    all_preds.sort(key=lambda x: x['score'], reverse=True)

    # Count total ground truth boxes for this class
    total_gt = 0
    gt_matched = {}  # image_id -> list of booleans (matched or not)

    for img_idx, gt in enumerate(ground_truths):
        labels = gt['labels']
        mask = labels == class_id
        num_gt = mask.sum().item()
        total_gt += num_gt
        gt_matched[img_idx] = [False] * num_gt

    # If no ground truth, return empty tensors
    if total_gt == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

    # Match predictions to ground truth
    tp = torch.zeros(len(all_preds))
    fp = torch.zeros(len(all_preds))

    for pred_idx, pred in enumerate(all_preds):
        img_idx = pred['image_id']
        pred_box = pred['box'].unsqueeze(0)  # [1, 4]

        gt = ground_truths[img_idx]
        gt_labels = gt['labels']
        gt_boxes = gt['boxes']

        # Get GT boxes for this class in this image
        mask = gt_labels == class_id
        if not mask.any():
            fp[pred_idx] = 1
            continue

        gt_boxes_class = gt_boxes[mask]  # [K, 4]

        # Calculate IoU with all GT boxes of this class
        iou = calculate_iou(pred_box, gt_boxes_class).squeeze(0)  # [K]

        # Find best matching GT
        max_iou, max_idx = iou.max(dim=0)

        if max_iou >= iou_threshold:
            if not gt_matched[img_idx][max_idx.item()]:
                tp[pred_idx] = 1
                gt_matched[img_idx][max_idx.item()] = True
            else:
                fp[pred_idx] = 1  # Already matched
        else:
            fp[pred_idx] = 1

    # Compute cumulative TP and FP
    cum_tp = torch.cumsum(tp, dim=0)
    cum_fp = torch.cumsum(fp, dim=0)

    # Precision and Recall
    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / total_gt

    # Confidence scores
    confidence_scores = torch.tensor([p['score'] for p in all_preds])

    return precision, recall, confidence_scores

def calculate_ap(precision: torch.Tensor, recall: torch.Tensor) -> float:
    """
    Calculate Average Precision (AP) using all-point interpolation (COCO standard).

    Args:
        precision: Precision array.
        recall: Recall array.

    Returns:
        Average Precision value.
    """
    if len(precision) == 0 or len(recall) == 0:
        return 0.0

    # Add sentinel values
    mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])

    # Compute the precision envelope (make precision monotonically decreasing)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = torch.max(mpre[i - 1], mpre[i])

    # Find points where recall changes
    i = torch.where(mrec[1:] != mrec[:-1])[0]

    # Sum areas under the curve
    ap = torch.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]).item()

    return ap

def calculate_map(
        predictions: List[Dict],
        ground_truths: List[Dict],
        iou_thresholds: List[float],
        num_classes: int,
        confidence_threshold: float = 0.3
) -> Dict[str, float]:
    """
    Calculate mean Average Precision (mAP) and global Precision/Recall/F1.

    Args:
        predictions: List of prediction dicts per image.
        ground_truths: List of ground truth dicts per image.
        iou_thresholds: List of IoU thresholds (e.g., [0.5] or [0.5:0.95]).
        num_classes: Number of classes (1-based, e.g., 7 for KITTI).
        confidence_threshold: Threshold to filter predictions for global P/R/F1.

    Returns:
        Dictionary with metrics.
    """
    metrics = {}

    # 1. Calculate mAP for each threshold
    ap_per_threshold = []
    for iou_thresh in iou_thresholds:
        ap_per_class = []
        for class_id in range(1, num_classes + 1):
            precision, recall, _ = compute_precision_recall(
                predictions, ground_truths, class_id, iou_thresh
            )
            ap = calculate_ap(precision, recall)
            ap_per_class.append(ap)

        mAP = sum(ap_per_class) / len(ap_per_class) if ap_per_class else 0.0
        ap_per_threshold.append(mAP)

        if iou_thresh == 0.5:
            metrics['mAP@0.5'] = mAP

    # Average mAP across all thresholds (COCO style mAP@0.5:0.95)
    metrics['mAP@0.5:0.95'] = sum(ap_per_threshold) / len(ap_per_threshold) if ap_per_threshold else 0.0

    # 2. Calculate global Precision, Recall, F1 at IoU=0.5 and fixed confidence threshold
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        pred_scores = pred['scores']

        gt_boxes = gt['boxes']
        gt_labels = gt['labels']

        for class_id in range(1, num_classes + 1):
            # Get predictions and GTs for current class
            p_mask = pred_labels == class_id
            g_mask = gt_labels == class_id

            p_boxes = pred_boxes[p_mask]
            p_scores = pred_scores[p_mask]
            g_boxes = gt_boxes[g_mask]

            num_preds = len(p_boxes)
            num_gts = len(g_boxes)

            if num_preds == 0 and num_gts == 0:
                continue

            if num_preds == 0:
                total_fn += num_gts
                continue

            if num_gts == 0:
                # Все предсказания - false positives (но только если confidence >= threshold)
                high_conf_mask = p_scores >= confidence_threshold
                total_fp += high_conf_mask.sum().item()
                continue

            # Фильтруем предсказания по confidence threshold
            conf_mask = p_scores >= confidence_threshold
            p_boxes_filtered = p_boxes[conf_mask]
            p_scores_filtered = p_scores[conf_mask]
            
            num_preds_filtered = len(p_boxes_filtered)
            
            if num_preds_filtered == 0:
                # Нет предсказаний с достаточной уверенностью - все GT это false negatives
                total_fn += num_gts
                continue

            # Calculate IoU matrix
            iou_matrix = calculate_iou(p_boxes_filtered, g_boxes)  # [num_preds_filtered, num_gts]

            # Match predictions to GTs greedily
            matched_gt_indices = set()
            tp = 0
            fp = 0

            # Sort predictions by score (descending) for greedy matching
            _, sorted_indices = p_scores_filtered.sort(descending=True)

            for idx in sorted_indices:
                best_iou, best_gt_idx = iou_matrix[idx].max(dim=0)

                if best_iou >= 0.5 and best_gt_idx.item() not in matched_gt_indices:
                    tp += 1
                    matched_gt_indices.add(best_gt_idx.item())
                else:
                    fp += 1

            fn = num_gts - tp

            total_tp += tp
            total_fp += fp
            total_fn += fn

    # Calculate final global metrics
    global_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    global_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall) if (
        global_precision + global_recall) > 0 else 0.0

    metrics['precision@0.5'] = global_precision
    metrics['recall@0.5'] = global_recall
    metrics['f1@0.5'] = global_f1

    return metrics

def evaluate_model(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        iou_thresholds: List[float],
        model_type: str = 'faster_rcnn',
        num_classes: int = 7
) -> Dict[str, float]:
    """
    Evaluate model on the validation set and compute all metrics.

    Args:
        model: Detection model.
        dataloader: Validation DataLoader.
        device: Device to evaluate on.
        iou_thresholds: List of IoU thresholds.
        model_type: Type of model ('faster_rcnn', 'ssd', 'efficientdet', 'detr').
        num_classes: Number of classes (1-based, e.g., 7 for KITTI).

    Returns:
        Dictionary with evaluation metrics.
    """
    model.eval()

    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = [image.to(device) for image in images]

            # Adjust targets for DETR
            if model_type == 'detr':
                targets = prepare_targets_for_detr(targets)

            # Forward pass (inference mode)
            predictions = model(images)

            # Store predictions and ground truths
            for pred, gt in zip(predictions, targets):
                # For DETR, shift labels back to 1-based indexing
                if model_type == 'detr':
                    pred['labels'] = pred['labels'] + 1

                all_predictions.append({
                    'boxes': pred['boxes'].cpu(),
                    'labels': pred['labels'].cpu(),
                    'scores': pred['scores'].cpu()
                })

                all_ground_truths.append({
                    'boxes': gt['boxes'].cpu(),
                    'labels': gt['labels'].cpu()
                })

    # Calculate metrics
    metrics = calculate_map(all_predictions, all_ground_truths, iou_thresholds, num_classes)

    return metrics