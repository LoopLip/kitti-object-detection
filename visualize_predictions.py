"""
Визуализация предсказаний всех моделей на тестовых изображениях.
Работает для Faster R-CNN, SSD, RetinaNet, DETR.
"""
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.dataset import KITTIDataset
from src.utils.utils import collate_fn, NUM_CLASSES, KITTI_CLASSES
from src.models import get_model

# Классы KITTI (1-based indexing для torchvision моделей)
CLASS_NAMES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Cyclist', 'Tram', 'Person_sitting']

# Цвета для классов (BGR формат для OpenCV)
COLORS_BGR = [
    (255, 0, 0),      # Car - красный
    (0, 255, 0),      # Van - зелёный
    (0, 0, 255),      # Truck - синий
    (255, 255, 0),    # Pedestrian - жёлтый
    (255, 0, 255),    # Cyclist - пурпурный
    (0, 255, 255),    # Tram - циан
    (128, 128, 128),  # Person_sitting - серый
]

# Цвета для matplotlib (RGB формат)
COLORS_RGB = [(c[2], c[1], c[0]) for c in COLORS_BGR]


def draw_predictions(image, boxes, labels, scores, min_confidence=0.3):
    """Рисует bounding boxes на изображении."""
    img = image.copy()
    
    for box, label, score in zip(boxes, labels, scores):
        if score < min_confidence:
            continue
        
        x1, y1, x2, y2 = [int(x) for x in box]
        
        # Проверяем границы
        h, w = img.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        # Получаем имя класса
        if 0 < label <= len(CLASS_NAMES):
            class_name = CLASS_NAMES[label - 1]
        else:
            class_name = f"Class_{label}"
        
        # Получаем цвет
        if 0 < label <= len(COLORS_BGR):
            color = COLORS_BGR[label - 1]
        else:
            color = (128, 128, 128)
        
        # Рисуем рамку
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # Рисуем подпись
        label_text = f"{class_name}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Фон для текста
        text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
        cv2.rectangle(img, (x1, y1 - text_size[1] - 10), 
                      (x1 + text_size[0], y1), color, -1)
        
        # Текст
        cv2.putText(img, label_text, (x1, y1 - 5), 
                    font, font_scale, (255, 255, 255), thickness)
    
    return img


def get_predictions(model, image, device, model_key, num_classes):
    """Получает предсказания от модели."""
    model.eval()
    
    # Для DETR передаём список тензоров, для остальных - batch
    if model_key == 'detr':
        image_input = [image.to(device)]
    else:
        image_input = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(image_input)
    
    # Извлекаем первое предсказание
    pred = predictions[0]
    
    # Получаем boxes, labels, scores
    boxes = pred['boxes'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    
    return boxes, labels, scores


def main():
    """Главная функция."""
    os.makedirs('results/plots', exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Загружаем валидационный датасет
    print("Loading validation dataset...")
    val_dataset = KITTIDataset(root_dir='data/raw/KITTI', split='val')
    print(f"Loaded {len(val_dataset)} images")
    
    # Модели для визуализации
    models_config = [
        ('Faster R-CNN', 'faster_rcnn', NUM_CLASSES + 1),
        ('SSD', 'ssd', NUM_CLASSES + 1),
        ('RetinaNet', 'efficientdet', NUM_CLASSES + 1),
        ('DETR', 'detr', NUM_CLASSES),
    ]
    
    # Проверяем наличие чекпоинтов
    available_models = []
    for model_name, model_key, num_classes in models_config:
        checkpoint_path = f"results/checkpoints/{model_key}_epoch_10.pth"
        if os.path.exists(checkpoint_path):
            available_models.append((model_name, model_key, num_classes))
            print(f"✅ {model_name}: чекпоинт найден")
        else:
            print(f"⚠️ {model_name}: чекпоинт не найден")
    
    if not available_models:
        print("❌ Нет доступных моделей для визуализации!")
        return
    
    # Берём 5 случайных изображений с объектами
    print("\nПоиск изображений с объектами...")
    images_with_objects = []
    
    for i in range(len(val_dataset)):
        image, target = val_dataset[i]
        if len(target['boxes']) > 0:
            images_with_objects.append(i)
        if len(images_with_objects) >= 10:
            break
    
    # Выбираем 5 случайных
    if len(images_with_objects) >= 5:
        indices = np.random.choice(images_with_objects, 5, replace=False)
    else:
        indices = images_with_objects[:5]
    
    print(f"Выбрано {len(indices)} изображений для визуализации\n")
    
    # Визуализация для каждого изображения
    for img_idx in indices:
        print(f"Обработка изображения {img_idx}...")
        image, target = val_dataset[img_idx]
        
        # Создаём фигуру
        num_cols = len(available_models) + 1  # +1 для Ground Truth
        fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 6))
        
        # Ground Truth
        img_np = image.permute(1, 2, 0).numpy() * 255
        img_np = img_np.astype(np.uint8)
        
        # Конвертируем RGB -> BGR для OpenCV
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Рисуем GT
        img_gt = draw_predictions(img_bgr, target['boxes'].numpy(), 
                                   target['labels'].numpy(), 
                                   np.ones(len(target['boxes'])))
        
        # Конвертируем обратно в RGB для matplotlib
        img_gt_rgb = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        
        axes[0].imshow(img_gt_rgb)
        axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Предсказания моделей
        for col_idx, (model_name, model_key, num_classes) in enumerate(available_models, start=1):
            checkpoint_path = f"results/checkpoints/{model_key}_epoch_10.pth"
            
            try:
                # Загружаем модель
                model = get_model(model_key, num_classes=num_classes, pretrained=False).to(device)
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # Получаем предсказания
                boxes, labels, scores = get_predictions(model, image, device, model_key, num_classes)
                
                # Рисуем предсказания
                img_pred = draw_predictions(img_bgr.copy(), boxes, labels, scores, min_confidence=0.3)
                img_pred_rgb = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)
                
                axes[col_idx].imshow(img_pred_rgb)
                axes[col_idx].set_title(f'{model_name}\n{len(boxes)} objects', fontsize=12)
                axes[col_idx].axis('off')
                
                print(f"  ✅ {model_name}: {len(boxes)} объектов")
                
            except Exception as e:
                print(f"  ❌ {model_name}: ошибка - {e}")
                axes[col_idx].text(0.5, 0.5, f'Error\n{str(e)[:50]}', 
                                  ha='center', va='center', fontsize=10,
                                  transform=axes[col_idx].transAxes)
                axes[col_idx].axis('off')
        
        # Сохраняем
        output_path = f'results/plots/predictions_sample_{img_idx}.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Сохранено: {output_path}\n")
        plt.close()
    
    print("=" * 60)
    print("✅ Визуализация завершена!")
    print(f" Результаты сохранены в: results/plots/")
    print("=" * 60)


if __name__ == "__main__":
    main()