import os
import random

# Папка с лейблами
label_dir = "data/raw/KITTI/label_2"

# Получаем все ID файлов
all_ids = sorted([
    f.replace(".txt", "") 
    for f in os.listdir(label_dir) 
    if f.endswith(".txt")
])

print(f"Найдено изображений с аннотациями: {len(all_ids)}")

# Перемешиваем и делим 80/20
random.seed(42)
random.shuffle(all_ids)

split_point = int(len(all_ids) * 0.8)
train_ids = all_ids[:split_point]
val_ids = all_ids[split_point:]

print(f"Train: {len(train_ids)} | Val: {len(val_ids)}")

# Создаём папку ImageSets
os.makedirs("data/raw/KITTI/ImageSets", exist_ok=True)

# Сохраняем train.txt
with open("data/raw/KITTI/ImageSets/train.txt", "w") as f:
    f.write("\n".join(train_ids))

# Сохраняем val.txt
with open("data/raw/KITTI/ImageSets/val.txt", "w") as f:
    f.write("\n".join(val_ids))

print("✅ Готово! Файлы train.txt и val.txt созданы.")