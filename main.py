import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn, optim
from sklearn.metrics import confusion_matrix

from utils.transforms import get_transforms
from utils.logger import Logger
from src.dataset import load_datasets
from src.model import get_model
from src.train import train_one_epoch
from src.evaluate import evaluate


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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
model = get_model(len(train_dataset.classes), device)


# 4. loss
all_labels = []

for _, y in train_loader:
    all_labels.extend(y.numpy())

all_labels = torch.tensor(all_labels)

counts = torch.bincount(all_labels, minlength=len(train_dataset.classes))
weights = counts.max() / (counts.float() + 1e-6)
weights = weights.to(device)

criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=4
)


# =========================
# 🔥 LOGGER
# =========================
logger = Logger()


# 5. training loop
model_path = "/Users/lomash/programmingPy/practice_2026/praktika_emoji/models/best_model.pth"
best_f1 = 0

if os.path.exists(model_path):
    print("Найдена предобученная модель! Загружаем веса...")
    model.load_state_dict(torch.load(model_path, map_location=device))

else:
    print("Предобученная модель не найдена. Начинаем обучение...")

    for epoch in range(80):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc, val_f1 = evaluate(
            model, test_loader, criterion, device
        )

        scheduler.step(val_f1)

        current_lr = optimizer.param_groups[0]['lr']

        # =========================
        # 🔥 LOGGER LOGGING
        # =========================
        logger.log(
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_f1,
            current_lr
        )

        print(f"Epoch {epoch + 1}")
        print(f"Train: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val:   {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        print(f"LR:    {current_lr}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print("🌟 Saved best model!")

        print("-" * 30)


# =========================
# 📊 ГРАФИКИ
# =========================
logger.plot_all()


# =========================
# CONFUSION MATRIX
# =========================
def get_predictions_and_targets(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    return all_targets, all_preds


def plot_confusion_matrix(targets, preds, class_names):
    cm = confusion_matrix(targets, preds, normalize='true')

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")

    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()


classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
targets, preds = get_predictions_and_targets(model, test_loader, device)

plot_confusion_matrix(targets, preds, class_names=classes)