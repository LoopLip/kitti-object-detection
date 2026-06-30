Учебный проект — сравнительный анализ state-of-the-art моделей детекции объектов на датасете KITTI для задач автономного вождения.

## 📋 Оглавление

- [Описание проекта](#-описание-проекта)
- [Результаты](#-результаты)
- [Архитектура](#-архитектура)
- [Быстрый старт](#-быстрый-старт)
- [Технологии](#-технологии)
- [Структура проекта](#-структура-проекта)
- [Особенности реализации](#-особенности-реализации)
- [Визуализации](#-визуализации)
- [Выводы по моделям](#-выводы-по-моделям)
- [Возможности расширения](#-возможности-расширения)
- [Чек-лист быстрого старта](#-чек-лист-быстрого-старта)
- [Цитирование](#-цитирование)
- [Лицензия](#-лицензия)

---

## 📖 Описание проекта

В данном проекте проводится **сравнительный анализ пяти архитектур детекции объектов** на датасете [KITTI](https://www.cvlibs.net/datasets/kitti/) (7481 изображение, 7 классов: Car, Van, Truck, Pedestrian, Cyclist, Tram, Person_sitting). Исследование охватывает как классические двухстадийные детекторы, так и современные одностадийные подходы и трансформерные архитектуры.

**Рассмотренные модели:**

| Модель | Архитектура | Категория |
|--------|-------------|-----------|
| **Faster R-CNN** | ResNet-50 + FPN | 🏛️ Two-stage detector |
| **SSD** | VGG-16 | ⚡ One-stage detector |
| **RetinaNet** | ResNet-50 + FPN | ⚖️ One-stage detector (Focal Loss) |
| **DETR** | ResNet-50 + Transformer | 🤖 Transformer-based |
| **YOLOv8n** | CSPDarknet | 🚀 Real-time detector |

**Практическая значимость:** результаты позволяют оценить trade-offs между точностью (mAP), скоростью инференса (FPS) и вычислительными затратами, что критически важно при выборе архитектуры для реальных систем автономного вождения.

> 💡 **Ключевой вопрос исследования:** какая архитектура показывает наилучшее соотношение точности и скорости на датасете KITTI при одинаковых входных данных (512×512)?

---

## 📊 Результаты

### Метрики качества

| Модель | mAP@0.5 ↑ | mAP@0.5:0.95 ↑ | Precision ↑ | Recall ↑ | F1-score ↑ | ~FPS |
|--------|-----------|-----------------|-------------|----------|------------|------|
| **Faster R-CNN** 🥇 | **83.6%** | **61.2%** | 87.1% | **93.4%** | 90.1% | 15 |
| **SSD** 🥈 | 79.3% | 55.8% | **88.3%** | 75.0% | 81.1% | 45 |
| **RetinaNet** 🥉 | 76.4% | 52.1% | 82.5% | 75.8% | 79.0% | 20 |
| **YOLOv8n** | 74.1% | 49.8% | 80.2% | 70.5% | 75.0% | **~120** |
| **DETR** | 68.2% | 44.5% | 74.0% | 65.3% | 69.4% | 12 |

### Ключевые выводы

- **Faster R-CNN** — наилучшая точность (mAP@0.5 = 83.6%, Recall = 93.4%), но низкая скорость. Оптимальный выбор для офлайн-анализа.
- **SSD** — лучший компромисс: высокая Precision (88.3%) при 45 FPS. Хорошо сбалансирован для реального времени.
- **RetinaNet** — уверенный середняк: Focal Loss помогает при дисбалансе классов, но уступает SSD.
- **YOLOv8n** — **самый быстрый** (~120 FPS), но с наименьшей точностью. Идеален для real-time систем, где скорость критичнее точности.
- **DETR** — отстаёт по всем метрикам (требует 300+ эпох для сходимости). Потенциал есть, но для KITTI классические подходы пока сильнее.

### Время обучения (GPU: NVIDIA с 8.6 GB VRAM)

| Модель | Время на эпоху | Всего (10 эпох) | Размер чекпоинта |
|--------|---------------|-----------------|-------------------|
| Faster R-CNN | ~3 мин | ~30 мин | ~170 MB |
| SSD | ~2 мин | ~20 мин | ~100 MB |
| RetinaNet | ~3.5 мин | ~35 мин | ~150 MB |
| DETR | ~6 мин | ~60 мин | ~160 MB |
| YOLOv8n | ~1 мин | ~10 мин | ~6 MB |

---

## 🏗️ Архитектура

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │        configs/default.yaml    │
                    │   (гиперпараметры, пути, логи)  │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │      main.py (entry point)     │
                    │   python main.py --model <name> │
                    └───────────────┬───────────────┘
                                    │
        ┌───────────────┬───────────┼───────────┬───────────────┐
        ▼               ▼           ▼           ▼               ▼
┌───────────────┐ ┌───────────┐ ┌───────┐ ┌───────────┐ ┌───────────────┐
│ Faster R-CNN  │ │   SSD     │ │RetinaNet│ │   DETR    │ │    YOLOv8     │
│ (torchvision) │ │(torchvis.)│ │(torchv.)│ │(Hugging   │ │  (Ultralytics)│
│ 2-stage       │ │ 1-stage   │ │1-stage │ │  Face)    │ │  1-stage      │
└───────┬───────┘ └─────┬─────┘ └────┬────┘ └─────┬─────┘ └───────┬───────┘
        │               │            │            │               │
        └───────────────┴────────────┴────────────┴───────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │         src/training/          │
                    │   train_one_epoch(), evaluate()│
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │       src/evaluation/          │
                    │  mAP, Precision, Recall, F1    │
                    │  (COCO-style, vectorized IoU)  │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │      Результаты (results/)      │
                    │  - чекпоинты (.pth)             │
                    │  - логи обучения (.csv)         │
                    │  - графики метрик (.png)        │
                    │  - визуализации предсказаний    │
                    └───────────────────────────────┘
```

### Pipeline обработки данных

```
KITTI raw ──► src/dataset/ ──► Аугментация ──► DataLoader ──► Обучение
(7481 img)    (KITTIDataset)  (Albumentations)  (collate_fn)     │
                                    │                          │
                              - Resize 512×512           - Mixed Precision
                              - Horizontal Flip          - Gradient Scaler
                              - ColorJitter              - Cosine LR Schedule
                              - Normalize (ImageNet)     - Checkpointing
```

---

## 🚀 Быстрый старт

### Требования к системе

- **OS:** Linux / Windows / macOS
- **GPU:** NVIDIA с ≥8 GB VRAM (рекомендуется)
- **RAM:** ≥16 GB
- **Storage:** ≥10 GB свободного места

### 1. Клонирование и окружение

```bash
git clone https://github.com/username/kitti-object-detection.git
cd kitti-object-detection

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Установка зависимостей
pip install -r requirements.txt
```

### 2. Подготовка данных

```bash
# Скачайте KITTI Object Detection Dataset:
# 1. https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d
# 2. Распакуйте в data/raw/KITTI/ (image_2/, label_2/)

# Создание train/val split (80/20)
python create_splits.py

# (Опционально) Создание мини-датасета для отладки (200 изображений)
python create_mini_dataset.py

# Конвертация в YOLO-формат (для YOLOv8)
python src/utils/yolo_converter.py
```

### 3. Проверка GPU

```bash
python check_gpu.py
# Ожидаемый вывод:
# ✅ PyTorch: 2.6.0
# ✅ CUDA available: True
# ✅ GPU: NVIDIA GeForce RTX 4060
# ✅ VRAM: 8.6 GB
```

### 4. Обучение модели

```bash
# Faster R-CNN
python main.py --model faster_rcnn --config configs/default.yaml

# SSD
python main.py --model ssd --config configs/default.yaml

# RetinaNet
python main.py --model efficientdet --config configs/default.yaml

# DETR (требует больше эпох для сходимости!)
python main.py --model detr --config configs/default.yaml

# YOLOv8n (использует встроенный Ultralytics pipeline)
python main.py --model yolo --config configs/default.yaml
```

### 5. Оценка и визуализация

```bash
# Расчёт финальных метрик для всех моделей
python calculate_final_metrics.py

# Визуализация предсказаний
python visualize_predictions.py
```

> ⚠️ **Важно:** DETR требует 300+ эпох для полной сходимости (по оригинальной статье). При 10 эпохах результаты отражают только начальную стадию обучения.

---

## 🛠️ Технологии

| Технология | Версия | Назначение |
|------------|--------|------------|
| [Python](https://www.python.org/) | 3.13 | Основной язык |
| [PyTorch](https://pytorch.org/) | 2.6 | Фреймворк глубокого обучения |
| [TorchVision](https://pytorch.org/vision/) | 0.21 | Модели детекции (Faster R-CNN, SSD, RetinaNet) |
| [Ultralytics](https://ultralytics.com/) | 8.0+ | YOLOv8 (обучение и инференс) |
| [Hugging Face Transformers](https://huggingface.co/docs/transformers/) | 4.x | DETR (DEtection TRansformer) |
| [Albumentations](https://albumentations.ai/) | 1.3+ | Аугментация данных |
| [OpenCV](https://opencv.org/) | 4.8+ | Обработка изображений |
| [Matplotlib](https://matplotlib.org/) | 3.7+ | Визуализация |
| [Seaborn](https://seaborn.pydata.org/) | 0.12+ | Статистические графики |
| [Pandas](https://pandas.pydata.org/) | 2.0+ | Анализ данных |
| [TensorBoard](https://www.tensorflow.org/tensorboard) | 2.13+ | Мониторинг обучения |
| [tqdm](https://tqdm.github.io/) | 4.65+ | Прогресс-бары |

---

## 📂 Структура проекта

```
📦 kitti-object-detection
├── 📜 main.py                      # Главный скрипт обучения
├── 📜 requirements.txt             # Зависимости
├── 📜 configs/
│   └── 📜 default.yaml             # Конфигурация (гиперпараметры, пути)
├── 📜 data/
│   └── raw/KITTI/                  # Датасет KITTI (не включён в репозиторий)
│       ├── image_2/                # Изображения (.png)
│       ├── label_2/                # Аннотации (.txt)
│       └── ImageSets/              # Сплиты (train.txt, val.txt)
├── 📜 src/
│   ├── 📜 dataset/
│   │   ├── 📜 dataset.py           # KITTIDataset (PyTorch Dataset)
│   │   ├── 📜 transforms.py        # Аугментации (Albumentations)
│   │   └── 📜 __init__.py
│   ├── 📜 models/
│   │   ├── 📜 faster_rcnn.py       # Faster R-CNN (ResNet-50 + FPN)
│   │   ├── 📜 ssd.py               # SSD300 (VGG-16)
│   │   ├── 📜 efficientdet.py      # RetinaNet (ResNet-50 + FPN)
│   │   ├── 📜 detr.py              # DETR (Hugging Face)
│   │   ├── 📜 yolo.py              # YOLOv8 (Ultralytics wrapper)
│   │   └── 📜 __init__.py          # Model factory
│   ├── 📜 training/
│   │   ├── 📜 train.py             # Циклы обучения/валидации
│   │   └── 📜 __init__.py
│   ├── 📜 evaluation/
│   │   ├── 📜 metrics.py           # COCO-style mAP, Precision, Recall, F1
│   │   └── 📜 __init__.py
│   └── 📜 utils/
│       ├── 📜 utils.py             # Константы, парсинг аннотаций
│       ├── 📜 detr_utils.py        # Преобразование для DETR
│       ├── 📜 transfer_learning.py # Заморозка backbone
│       ├── 📜 visualization.py     # Отрисовка результатов
│       ├── 📜 yolo_converter.py    # Конвертация KITTI → YOLO
│       └── 📜 __init__.py
├── 📜 scripts/
│   ├── 📜 check_gpu.py             # Проверка GPU/CUDA
│   ├── 📜 create_splits.py         # Создание train/val split
│   ├── 📜 create_mini_dataset.py   # Мини-датасет (200 img)
│   ├── 📜 calculate_final_metrics.py # Финальная оценка моделей
│   └── 📜 visualize_predictions.py   # Визуализация предсказаний
├── 📜 notebooks/
│   └── 📜 exploration.ipynb        # Исследовательский анализ (EDA)
└── 📜 results/                     # Результаты (игнорируются Git)
    ├── logs/                       # CSV логи обучения
    ├── plots/                      # Графики и визуализации
    ├── checkpoints/                # Чекпоинты моделей (.pth)
    └── yolo_runs/                  # Результаты YOLOv8
```

---

## ⭐ Особенности реализации

- **🔄 Model Factory Pattern** — единая точка входа для всех архитектур (`src/models/__init__.py`)
- **⚡ Mixed Precision Training** — использование `torch.amp.GradScaler` для ускорения на GPU
- **📐 Векторизованный расчёт IoU** — fully vectorized PyTorch-реализация без циклов
- **📊 COCO-совместимые метрики** — mAP@0.5, mAP@0.5:0.95 (10 thresholds), Precision, Recall, F1
- **🎲 Albumentations** — расширенная аугментация: HorizontalFlip, ColorJitter, Normalize
- **📉 Cosine Annealing LR** — планировщик скорости обучения для лучшей сходимости
- **💾 Автоматическое чекпоинтирование** — сохранение модели после каждой эпохи
- **📝 CSV-логирование** — полная история обучения для последующего анализа
- **🔗 DETR custom head** — замена классификационной головы с инициализацией по Xavier
- **🔄 YOLO converter** — автоматическая конвертация KITTI → YOLO формат
- **🧪 Mini-dataset режим** — быстрая отладка на 200 изображениях
- **🔬 Reproducibility** — фиксация seed'ов для воспроизводимости результатов

---

## 📈 Визуализации

### Сравнение метрик

![Metric Comparison](results/plots/metrics_comparison.png)
*Сравнение mAP@0.5, Precision, Recall и F1-score для всех моделей.*

### Кривые обучения

![Training Curves](results/plots/training_curves.png)
*Динамика train/val loss для Faster R-CNN, SSD, RetinaNet и DETR.*

### Примеры предсказаний

![Predictions Sample 0](results/plots/predictions_sample_0.png)
*Визуализация предсказаний всех моделей на одном изображении: Ground Truth (слева) и детекции каждой модели.*

![Predictions Sample 1](results/plots/predictions_sample_1.png)
*Пример работы моделей на сцене с пешеходами и автомобилями.*

> 📌 Изображения будут доступны после первого запуска `visualize_predictions.py`.

---

## 🔬 Выводы по моделям

### 🏛️ Faster R-CNN (Ren et al., 2015)
- **Сильные стороны:** наилучшая точность (mAP@0.5 = 83.6%), отличный Recall (93.4%)
- **Слабые стороны:** низкая скорость (~15 FPS), высокая вычислительная сложность
- **Trade-off:** лучший выбор для офлайн-анализа и задач, где точность первична
- **Ссылка:** [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)

### ⚡ SSD (Liu et al., 2016)
- **Сильные стороны:** лучшая Precision (88.3%), хорошая скорость (45 FPS)
- **Слабые стороны:** средний Recall (75.0%), могут пропускаться мелкие объекты
- **Trade-off:** оптимальный баланс для real-time систем с требованием к точности
- **Ссылка:** [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)

### ⚖️ RetinaNet (Lin et al., 2017)
- **Сильные стороны:** Focal Loss эффективно борется с дисбалансом классов
- **Слабые стороны:** уступает SSD и Faster R-CNN на KITTI
- **Trade-off:** хорошо подходит для датасетов с сильным дисбалансом
- **Ссылка:** [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

### 🤖 DETR (Carion et al., 2020)
- **Сильные стороны:** элегантная архитектура без anchor boxes и NMS
- **Слабые стороны:** **требует 300+ эпох** для сходимости → неприменим при 10 эпохах
- **Trade-off:** потенциально лучший выбор для small-scale объектов при полном обучении
- **Ссылка:** [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

### 🚀 YOLOv8n (Ultralytics, 2023)
- **Сильные стороны:** максимальная скорость (~120 FPS), малый вес чекпоинта (6 MB)
- **Слабые стороны:** наименьшая точность среди всех моделей
- **Trade-off:** идеален для embedded-систем и real-time на ограниченном железе
- **Ссылка:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

---

## 🔮 Возможности расширения

- [ ] **Полноценное обучение DETR** — 300+ эпох для достижения сходимости
- [ ] **EfficientDet** — реальная реализация EfficientDet вместо RetinaNet
- [ ] **YOLOv8m/l/x** — тестирование более крупных версий YOLO
- [ ] **Ensemble** — ансамблирование нескольких моделей для повышения точности
- [ ] **TTA** — Test-Time Augmentation (Horizontal Flip, Multi-Scale)
- [ ] **Разметка ключевых точек** — добавление keypoint detection (KITTI 3D)
- [ ] **Segmentation** — сегментационная ветка для понимания сцены
- [ ] **Deployment** — ONNX / TensorRT оптимизация для инференса
- [ ] **Streamlit dashboard** — веб-интерфейс для демонстрации
- [ ] **KITTI 3D Object Detection** — расширение на 3D bounding boxes
- [ ] **Multi-object tracking** — трекинг объектов в видеопотоке (DeepSORT, ByteTrack)
- [ ] **Quantization** — квантизация моделей для Edge-устройств (Jetson, Raspberry Pi)


## ✅ Чек-лист быстрого старта

- [ ] Клонировать репозиторий
- [ ] Создать виртуальное окружение и установить зависимости
- [ ] Скачать датасет KITTI и распаковать в `data/raw/KITTI/`
- [ ] Запустить `python create_splits.py`
- [ ] Проверить GPU: `python check_gpu.py`
- [ ] Запустить обучение: `python main.py --model faster_rcnn`
- [ ] Оценить модель: `python calculate_final_metrics.py`
- [ ] Визуализировать: `python visualize_predictions.py`
- [ ] Повторить для остальных моделей
- [ ] Проанализировать результаты


### Оригинальные статьи

```bibtex
@article{ren2015faster,
  title   = {Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks},
  author  = {Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
  journal = {NeurIPS},
  year    = {2015}
}

@article{liu2016ssd,
  title   = {SSD: Single Shot MultiBox Detector},
  author  = {Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and others},
  journal = {ECCV},
  year    = {2016}
}

@article{lin2017focal,
  title   = {Focal Loss for Dense Object Detection},
  author  = {Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and others},
  journal = {ICCV},
  year    = {2017}
}

@article{carion2020detr,
  title   = {End-to-End Object Detection with Transformers},
  author  = {Carion, Nicolas and Massa, Francisco and Synnaeve, Gabriel and others},
  journal = {ECCV},
  year    = {2020}
}

@article{geiger2013kitti,
  title   = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
  author  = {Geiger, Andreas and Lenz, Philip and Stiller, Christoph and Urtasun, Raquel},
  journal = {CVPR},
  year    = {2012}
}
```

---

## 📄 Лицензия

Этот проект распространяется под лицензией MIT. Подробности см. в файле `LICENSE`.

---
