import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()

        # 1-й сверточный блок
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # 2-й сверточный блок
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Полносвязная часть
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 48 → 24
        x = self.pool(F.relu(self.conv2(x)))   # 24 → 12

        x = x.view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def get_model2(num_classes, device):
    # Создаем экземпляр нашей кастомной сети
    model = SimpleCNN(num_classes=num_classes)

    # Отправляем модель на нужное устройство (CPU, GPU или MPS)
    model = model.to(device)

    return model