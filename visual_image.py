import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image

# --- Настройка путей ---
# Укажите путь к РАСПАКОВАННОЙ папке train
TRAIN_DIR = "/Users/lomash/programmingPy/practice_2026/archive/train" 
NUM_IMAGES_PER_CLASS = 4

# 1. Сбор информации из папок
class_names = []
class_counts = {}

if not os.path.exists(TRAIN_DIR):
    print(f"Ошибка: Папка '{TRAIN_DIR}' не найдена. Укажите правильный путь.")
    exit()

for class_name in sorted(os.listdir(TRAIN_DIR)):
    class_path = os.path.join(TRAIN_DIR, class_name)
    
    if os.path.isdir(class_path):
        class_names.append(class_name)
        # Считаем только файлы изображений
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        class_counts[class_name] = len(images)

# 2. График распределения
df_counts = pd.DataFrame(list(class_counts.items()), columns=['Emotion', 'Count']).sort_values(by='Count', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x='Count', y='Emotion', data=df_counts, palette='flare')
plt.title('Распределение эмоций в папке Train (FER-2013)')
plt.xlabel('Количество картинок')
plt.ylabel('Эмоция')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 3. Вывод примеров лиц
num_classes = len(class_names)
fig, axes = plt.subplots(num_classes, NUM_IMAGES_PER_CLASS, figsize=(NUM_IMAGES_PER_CLASS * 2, num_classes * 2))

for i, class_name in enumerate(class_names):
    class_path = os.path.join(TRAIN_DIR, class_name)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    sampled_images = random.sample(images, min(NUM_IMAGES_PER_CLASS, len(images)))
    
    for j in range(NUM_IMAGES_PER_CLASS):
        ax = axes[i, j]
        if j < len(sampled_images):
            img_path = os.path.join(class_path, sampled_images[j])
            img = Image.open(img_path).convert('L') # 'L' означает grayscale (черно-белое)
            ax.imshow(img, cmap='gray')
        ax.axis('off')
        
        if j == 0:
            ax.set_ylabel(class_name, rotation=0, size='large', labelpad=40)
            ax.get_yaxis().set_visible(True)
            ax.set_yticks([])

plt.tight_layout()
plt.subplots_adjust(left=0.2)
plt.show()