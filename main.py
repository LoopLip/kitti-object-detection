"""
Main entry point for the CV project.
Usage: python main.py --model faster_rcnn --config configs/default.yaml
"""

import os
import csv
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler

from src.dataset import KITTIDataset
from src.utils.utils import collate_fn, NUM_CLASSES
from src.models import get_model
from src.training.train import train_one_epoch, evaluate
from src.utils.detr_utils import prepare_targets_for_detr
from src.utils.yolo_converter import convert_kitti_to_yolo_format
from src.utils.transfer_learning import freeze_backbone


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    """Main execution function."""
    # 1. Parse arguments
    parser = argparse.ArgumentParser(description="CV Project Training Pipeline")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()

    # 2. Load configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    set_seed(config['reproducibility']['seed'])
    
    # Ускорение на GPU
    torch.backends.cudnn.benchmark = config['reproducibility'].get('benchmark', False)

    # 3. Setup device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 4. Create directories
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)

    # --- YOLO Specific Branch ---
    if args.model == 'yolo':
        print("Detected YOLO model. Bypassing custom training loop.")

        # Convert data if not already done
        yolo_data_dir = config['yolo']['data_dir']
        data_yaml = os.path.join(yolo_data_dir, 'data.yaml')

        if not os.path.exists(data_yaml):
            print("Converting KITTI to YOLO format...")
            convert_kitti_to_yolo_format(config['data']['root_dir'], yolo_data_dir, 'train')
            convert_kitti_to_yolo_format(config['data']['root_dir'], yolo_data_dir, 'val')
        else:
            print(f"YOLO data already exists at {data_yaml}")

        # Initialize YOLO Wrapper
        model = get_model('yolo', num_classes=NUM_CLASSES)

        # Run native Ultralytics training
        model.train_pipeline(
            data_yaml=data_yaml,
            epochs=config['training']['num_epochs'],
            imgsz=config['yolo']['imgsz'],
            batch=config['yolo']['batch'],
            device=device
        )
        print("YOLO training completed successfully!")
        return

    # --- Standard PyTorch Branch (Faster R-CNN, SSD, EfficientDet, DETR) ---

    # 5. Initialize Datasets and DataLoaders
    train_dataset = KITTIDataset(
        root_dir=config['data']['root_dir'],
        split=config['data']['train_split']
    )
    val_dataset = KITTIDataset(
        root_dir=config['data']['root_dir'],
        split=config['data']['val_split']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=False
    )

    # 6. Initialize Model
    # Note: DETR does not include background class, others do.
    num_classes = NUM_CLASSES if args.model == 'detr' else (NUM_CLASSES + 1)

    model = get_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=config['model']['pretrained']
    ).to(device)

    # 7. Initialize Optimizer and Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    
    # DETR-specific: AdamW + lower LR (как в оригинальной статье)
    if args.model == 'detr':
        optimizer = optim.AdamW(
            params,
            lr=0.0001,
            weight_decay=1e-4
        )
        print("⚡ DETR: Using AdamW optimizer with lr=0.0001")
    else:
        if config['training']['optimizer'] == 'SGD':
            optimizer = optim.SGD(
                params,
                lr=config['training']['learning_rate'],
                momentum=config['training']['momentum'],
                weight_decay=config['training']['weight_decay']
            )
        else:
            optimizer = optim.Adam(params, lr=config['training']['learning_rate'])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs']
    )

    # 8. Initialize Mixed Precision Scaler
    scaler = GradScaler()

    # 9. Setup CSV Logging
    log_file_path = os.path.join(config['logging']['log_dir'], f"{args.model}_training_log.csv")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    with open(log_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr'])

        # 10. Training Loop
        print(f"Starting training for {config['training']['num_epochs']} epochs...")
        for epoch in range(1, config['training']['num_epochs'] + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, device, epoch, scaler,
                model_type=args.model
            )
            val_loss = evaluate(
                model, val_loader, device, epoch,
                model_type=args.model
            )

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # Write logs to CSV
            csv_writer.writerow([epoch, train_loss, val_loss, current_lr])

            print(f"Epoch {epoch}/{config['training']['num_epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"LR: {current_lr:.6f}")

            # 11. Save Checkpoint
            checkpoint_path = os.path.join(
                config['logging']['checkpoint_dir'],
                f"{args.model}_epoch_{epoch}.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)

    print(f"Training completed successfully! Logs saved to {log_file_path}")


if __name__ == "__main__":
    main()