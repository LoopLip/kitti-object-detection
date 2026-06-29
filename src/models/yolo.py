import torch
from ultralytics import YOLO


class YOLOWrapper(torch.nn.Module):
    """
    Wrapper for Ultralytics YOLO model to fit the project's model factory pattern.
    """

    def __init__(self, model_name: str = 'yolov8n.pt', num_classes: int = 7) -> None:
        """
        Initialize YOLO Wrapper.

        Args:
            model_name: Pretrained YOLO weights file (e.g., 'yolov8n.pt').
            num_classes: Number of classes (Ultralytics handles this via data.yaml).
        """
        super().__init__()
        self.model = YOLO(model_name)
        self.num_classes = num_classes

    def train_pipeline(self, data_yaml: str, epochs: int, imgsz: int = 640, batch: int = 16, **kwargs) -> None:
        """
        Execute the native Ultralytics training pipeline.
        This method should be called directly from main.py if model == 'yolo'.

        Args:
            data_yaml: Path to the KITTI data.yaml file for Ultralytics.
            epochs: Number of training epochs.
            imgsz: Input image size.
            batch: Batch size.
            **kwargs: Additional arguments for YOLO train().
        """
        self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project='results/yolo_runs',
            name='yolov8_kitti',
            **kwargs
        )

    def forward(self, images: torch.Tensor, targets: list = None) -> list:
        """
        Forward pass for inference.
        Note: Training forward pass is handled internally by Ultralytics.

        Args:
            images: Input images tensor.
            targets: Ignored for YOLO inference.

        Returns:
            List of prediction dictionaries.
        """
        # Ultralytics expects numpy arrays or PIL images for predict(),
        # but for tensor inference, we use the underlying predictor.
        results = self.model.predict(source=images, verbose=False)
        return results