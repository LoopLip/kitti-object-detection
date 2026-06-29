import torch
import torch.nn as nn
from transformers import DetrForObjectDetection, DetrImageProcessor


class DETRWrapper(nn.Module):
    """Stable DETR wrapper."""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        
        if pretrained:
            self.model = DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50",
                ignore_mismatched_sizes=True
            )
            
            # Обновляем конфиг
            self.model.config.num_labels = num_classes
            
            # Создаём НОВЫЙ classification head с правильным размером
            # Выход: num_classes + 1 (последний класс = no-object)
            d_model = self.model.config.d_model
            new_classifier = nn.Linear(d_model, num_classes + 1)
            
            # Инициализация как в оригинальном DETR
            nn.init.normal_(new_classifier.weight, std=0.01)
            nn.init.constant_(new_classifier.bias, 0)
            
            # ЗАМЕНЯЕМ head в модели
            self.model.class_labels_classifier = new_classifier
            
            # ВАЖНО: empty_weight должен быть обычным тензором (не buffer)
            # и должен быть на том же устройстве что и модель
            empty_weight = torch.ones(num_classes + 1)
            empty_weight[-1] = 0.1  # Меньший вес для no-object class
            self.model.empty_weight = empty_weight
            
        else:
            from transformers import DetrConfig
            config = DetrConfig(num_labels=num_classes)
            self.model = DetrForObjectDetection(config)
        
        self.num_classes = num_classes
        self.debug_counter = 0
    
    def forward(self, images, targets=None):
        """Forward pass."""
        pixel_values = torch.stack(images)
        
        # Перемещаем empty_weight на правильное устройство
        if hasattr(self.model, 'empty_weight'):
            self.model.empty_weight = self.model.empty_weight.to(pixel_values.device)
        
        if targets is not None and self.training:
            batch_size = len(images)
            labels = []
            
            for i in range(batch_size):
                target = targets[i]
                boxes = target['boxes'].clone()
                class_labels = target['labels'].clone()
                
                h, w = images[i].shape[1], images[i].shape[2]
                
                # xyxy -> cxcywh (нормализованный 0-1)
                x_min = boxes[:, 0].float() / w
                y_min = boxes[:, 1].float() / h
                x_max = boxes[:, 2].float() / w
                y_max = boxes[:, 3].float() / h
                
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                w_box = x_max - x_min
                h_box = y_max - y_min
                
                boxes_cxcywh = torch.stack([cx, cy, w_box, h_box], dim=1)
                boxes_cxcywh = torch.clamp(boxes_cxcywh, min=0.0, max=1.0)
                
                labels.append({
                    "boxes": boxes_cxcywh,
                    "class_labels": class_labels
                })
            
            if self.debug_counter == 0:
                print(f"\n DETR Training Debug:")
                print(f"   Batch size: {batch_size}")
                print(f"   Total objects: {sum(len(l['boxes']) for l in labels)}")
                print(f"   Boxes range: [{min(l['boxes'].min().item() for l in labels):.4f}, {max(l['boxes'].max().item() for l in labels):.4f}]")
                print(f"   Labels range: [{min(l['class_labels'].min().item() for l in labels)}, {max(l['class_labels'].max().item() for l in labels)}]")
                print(f"   Num classes: {self.num_classes}")
                print(f"   Classifier output size: {self.model.class_labels_classifier.out_features}")
                print(f"   Empty weight size: {self.model.empty_weight.shape}")
                print()
            
            self.debug_counter += 1
            
            outputs = self.model(pixel_values=pixel_values, labels=labels)
            
            if torch.isnan(outputs.loss):
                return {'loss': torch.tensor(0.0, requires_grad=True, device=pixel_values.device)}
            
            if self.debug_counter <= 3:
                print(f"   DETR loss: {outputs.loss.item():.4f}")
            
            return {'loss': outputs.loss}
        
        else:
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)
            
            if torch.isnan(outputs.logits).any():
                return [{'boxes': torch.zeros((0, 4)), 'labels': torch.zeros(0, dtype=torch.long), 'scores': torch.zeros(0)}]
            
            target_sizes = [(img.shape[1], img.shape[2]) for img in images]
            
            processed_outputs = self.processor.post_process_object_detection(
                outputs,
                threshold=0.0,
                target_sizes=target_sizes
            )
            
            results = []
            for output in processed_outputs:
                results.append({
                    'boxes': output['boxes'],
                    'labels': output['labels'],
                    'scores': output['scores']
                })
            
            return results


def _get_detr(num_classes: int, pretrained: bool) -> nn.Module:
    print("Loading DETR from Hugging Face Transformers...")
    model = DETRWrapper(num_classes, pretrained)
    print(f"✅ DETR loaded successfully with {num_classes} classes")
    return model