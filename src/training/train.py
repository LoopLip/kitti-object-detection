"""
Training and evaluation loops for object detection models.
"""

from typing import Dict, List, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from src.utils.detr_utils import prepare_targets_for_detr


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: GradScaler,
    model_type: str = 'faster_rcnn'
) -> float:
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
    
    for images, targets in progress_bar:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        if model_type == 'detr':
            targets = prepare_targets_for_detr(targets)
        
        if any(len(t['boxes']) == 0 for t in targets):
            continue
        
        optimizer.zero_grad()
        
        # Для DETR отключаем mixed precision
        if model_type == 'detr':
            loss_dict = model(images, targets)
        else:
            with autocast(device_type='cuda', dtype=torch.float16):
                loss_dict = model(images, targets)
        
        # Extract loss
        loss = None
        if isinstance(loss_dict, dict):
            if 'loss' in loss_dict:
                loss = loss_dict['loss']
            else:
                loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))
        elif isinstance(loss_dict, torch.Tensor):
            loss = loss_dict
        
        if loss is None:
            continue
        
        if torch.isnan(loss):
            continue
        
        # Backward pass
        if model_type == 'detr':
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    model_type: str = 'faster_rcnn'
) -> float:
    """Evaluate the model on the validation set."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]  ", leave=False)
    
    with torch.no_grad():
        for images, targets in progress_bar:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            if model_type == 'detr':
                targets = prepare_targets_for_detr(targets)
            
            if any(len(t['boxes']) == 0 for t in targets):
                continue
            
            if model_type == 'detr':
                loss_dict = model(images, targets)
            else:
                with autocast(device_type='cuda', dtype=torch.float16):
                    loss_dict = model(images, targets)
            
            loss = None
            if isinstance(loss_dict, dict):
                if 'loss' in loss_dict:
                    loss = loss_dict['loss']
                else:
                    loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))
            elif isinstance(loss_dict, torch.Tensor):
                loss = loss_dict
            
            if loss is None or torch.isnan(loss):
                continue
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss