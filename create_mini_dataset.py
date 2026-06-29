import os
import shutil
import random

src_dir = "data/raw/KITTI"
dst_dir = "data/raw/KITTI_mini"

# Создаём структуру
os.makedirs(f"{dst_dir}/image_2", exist_ok=True)
os.makedirs(f"{dst_dir}/label_2", exist_ok=True)
os.makedirs(f"{dst_dir}/ImageSets", exist_ok=True)

# Читаем список всех ID
with open(f"{src_dir}/ImageSets/train.txt", 'r') as f:
    all_ids = [line.strip() for line in f.readlines()]

# Берём случайные 200 изображений
random.seed(42)
selected_ids = random.sample(all_ids, 200)

print(f"Копируем {len(selected_ids)} изображений...")

for img_id in selected_ids:
    shutil.copy(f"{src_dir}/image_2/{img_id}.png", f"{dst_dir}/image_2/{img_id}.png")
    shutil.copy(f"{src_dir}/label_2/{img_id}.txt", f"{dst_dir}/label_2/{img_id}.txt")

# Создаём train/val split (80/20)
random.shuffle(selected_ids)
split_point = int(len(selected_ids) * 0.8)

with open(f"{dst_dir}/ImageSets/train.txt", 'w') as f:
    f.write('\n'.join(selected_ids[:split_point]))
with open(f"{dst_dir}/ImageSets/val.txt", 'w') as f:
    f.write('\n'.join(selected_ids[split_point:]))

print(f"✅ Готово! Train: {split_point}, Val: {len(selected_ids)-split_point}")