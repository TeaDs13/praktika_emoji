import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn, optim
from sklearn.metrics import confusion_matrix

from utils.transforms import get_transforms
from src.dataset import load_datasets
from src.model import get_model
from src.train import train_one_epoch
from src.evaluate import evaluate

device = torch.device("cuda" if torch.backends.mps.is_available() else "cpu")

# 1. transforms
train_tf, test_tf = get_transforms()

# 2. dataset
train_dataset, test_dataset = load_datasets(
    "/Users/lomash/programmingPy/practice_2026/archive/train",
    "/Users/lomash/programmingPy/practice_2026/archive/test",
    train_tf,
    test_tf
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

# 3. model
# Изменяем: передаем device внутрь функции get_model
model = get_model(len(train_dataset.classes), device)

# 4. loss + optimizer + scheduler
# Взвешиваем классы: редким эмоциям (Disgust) даем вес больше, частым (Happy) - меньше
# Порядок FER2013: ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# Явно указываем dtype=torch.float32 и переносим на правильный device
class_weights = torch.FloatTensor([1.0, 9.0, 1.0, 0.5, 0.8, 1.2, 0.8]).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Добавляем планировщик: если Val Acc не растет 4 эпохи подряд, уменьшаем LR в 2 раза
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
# 5. training loop
model_path = "/Users/lomash/programmingPy/practice_2026/praktika_emoji/models/best_model.pth"

if os.path.exists(model_path):
    print("Найдена предобученная модель! Загружаем веса...")
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print("Предобученная модель не найдена. Начинаем обучение...")
    best_acc = 0

    for epoch in range(80   ):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        
        # ДЕЛАЕМ ШАГ ПЛАНИРОВЩИКОМ НА ОСНОВЕ ВАЛИДАЦИИ
        scheduler.step(val_acc)

        print(f"Epoch {epoch+1}")
        print(f"Train: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val:   {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Смотрим текущий Learning Rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr}")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print("🌟 Saved best model!")
        print("-" * 30)


def get_predictions_and_targets(model, loader, device):
    """Прогоняет модель по тестовым данным и собирает ответы."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            
            _, pred = outputs.max(1)

            # Переносим на CPU и превращаем в обычные массивы numpy
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    return all_targets, all_preds

def plot_confusion_matrix(targets, preds, class_names):
    """Строит и отображает красивую тепловую карту Confusion Matrix."""
    # 1. Вычисляем саму матрицу ошибок
    cm = confusion_matrix(targets, preds)
    
    # 2. Настраиваем размер окна графика
    plt.figure(figsize=(10, 8))
    
    # 3. Рисуем тепловую карту с помощью seaborn
    sns.heatmap(
        cm, 
        annot=True,          # Включает отображение цифр внутри квадратов
        fmt="d",             # Формат чисел (d - целые числа)
        cmap="Blues",        # Цвет карты (синие оттенки)
        xticklabels=class_names, # Названия классов по горизонтали (Предсказано)
        yticklabels=class_names  # Названия классов по вертикали (Реальность)
    )
    
    # 4. Добавляем подписи осей
    plt.title("Матрица ошибок (Confusion Matrix)", fontsize=16)
    plt.ylabel("Реальные эмоции", fontsize=12)
    plt.xlabel("Предсказанные моделью эмоции", fontsize=12)
    
    # Поворачиваем надписи, чтобы они не налезали друг на друга
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # 5. Сохраняем картинку на диск и показываем на экране
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()

# 1. Твои названия классов в правильном порядке
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 2. Собираем ответы (передаем туда твою модель, test_loader и устройство)
targets, preds = get_predictions_and_targets(model, test_loader, device)

# 3. Рисуем матрицу!
plot_confusion_matrix(targets, preds, class_names=classes)    