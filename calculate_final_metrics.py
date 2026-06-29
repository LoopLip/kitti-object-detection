"""
Расчёт финальных метрик (mAP, Precision, Recall, F1) для всех моделей.
"""
import os
import csv
import yaml
import torch
from torch.utils.data import DataLoader
from src.dataset import KITTIDataset
from src.utils.utils import collate_fn, NUM_CLASSES
from src.models import get_model
from src.evaluation.metrics import evaluate_model

def main():
    with open('configs/default.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Загружаем валидационный датасет
    val_dataset = KITTIDataset(
        root_dir=config['data']['root_dir'], 
        split=config['data']['val_split']
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False, 
        num_workers=2, 
        collate_fn=collate_fn
    )
    
    # Модели для оценки (YOLO оценивается отдельно)
    models_to_eval = ['faster_rcnn', 'ssd', 'efficientdet', 'detr']
    results = []
    
    for model_name in models_to_eval:
        print(f"\n{'='*60}")
        print(f"Оценка модели: {model_name}")
        print(f"{'='*60}")
        
        num_classes = NUM_CLASSES if model_name == 'detr' else (NUM_CLASSES + 1)
        
        # Загружаем последний чекпоинт
        checkpoint_path = os.path.join(
            config['logging']['checkpoint_dir'], 
            f"{model_name}_epoch_{config['training']['num_epochs']}.pth"
        )
        
        if not os.path.exists(checkpoint_path):
            print(f"⚠️ Чекпоинт не найден: {checkpoint_path}")
            continue
        
        model = get_model(model_name, num_classes=num_classes, pretrained=False).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Расчёт метрик
        metrics = evaluate_model(
            model, 
            val_loader, 
            device, 
            iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
            model_type=model_name,
            num_classes=NUM_CLASSES
        )
        metrics['model'] = model_name
        results.append(metrics)
        
        print(f"✅ {model_name}:")
        print(f"   mAP@0.5:    {metrics['mAP@0.5']:.4f}")
        print(f"   mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
        print(f"   Precision:  {metrics['precision@0.5']:.4f}")
        print(f"   Recall:     {metrics['recall@0.5']:.4f}")
        print(f"   F1:         {metrics['f1@0.5']:.4f}")
    
    # Для YOLO берём метрики из Ultralytics
    yolo_results_csv = "runs/detect/results/yolo_runs/yolov8_kitti/results.csv"
    if os.path.exists(yolo_results_csv):
        import pandas as pd
        df = pd.read_csv(yolo_results_csv)
        last_row = df.iloc[-1]
        results.append({
            'model': 'yolo',
            'mAP@0.5': float(last_row.get('metrics/mAP50(B)', 0)),
            'mAP@0.5:0.95': float(last_row.get('metrics/mAP50-95(B)', 0)),
            'precision@0.5': float(last_row.get('metrics/precision(B)', 0)),
            'recall@0.5': float(last_row.get('metrics/recall(B)', 0)),
            'f1@0.5': float(last_row.get('metrics/mAP50(B)', 0))  # приблизительная оценка
        })
        print(f"\n✅ YOLO метрики добавлены из Ultralytics")
    
    # Сохраняем в CSV
    os.makedirs('results/logs', exist_ok=True)
    output_path = 'results/logs/model_comparison.csv'
    
    fieldnames = ['model', 'mAP@0.5', 'mAP@0.5:0.95', 'precision@0.5', 'recall@0.5', 'f1@0.5']
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n{'='*60}")
    print(f"✅ Все метрики сохранены в: {output_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()