import torch.nn as nn

def freeze_backbone(model: nn.Module) -> None:
    """Замораживает backbone модели (все слои кроме head)."""
    if hasattr(model, 'backbone'):
        for param in model.backbone.parameters():
            param.requires_grad = False
        for module in model.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        print("✅ Backbone frozen")

def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)