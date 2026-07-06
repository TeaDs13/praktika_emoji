import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        
        # Блок 1: Вход (1, 48, 48) -> Выход (64, 24, 24)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(p=0.25)

        # Блок 2: Вход (64, 24, 24) -> Выход (128, 12. 12)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout2d(p=0.25)

        # Блок 3: Вход (128, 12, 12) -> Выход (256, 6, 6)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout2d(p=0.25)
        
        # Блок 4: Вход (256, 6, 6) -> Выход (512, 3, 3)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout2d(p=0.25)

        # Полносвязные слои (Classifiers)
        # После 4-х пулингов (размер уменьшается в 2^4 = 16 раз): 48 / 16 = 3. 
        # Размер тензора перед Flatten: 512 каналов * 3 * 3
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 3 * 3, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.drop_fc1 = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.drop_fc2 = nn.Dropout(p=0.5)
        
        # Выходной слой на 7 классов (без Softmax, так как используется CrossEntropyLoss)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Проход через сверточные блоки
        x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        x = self.drop4(self.pool4(F.relu(self.bn4(self.conv4(x)))))
        
        # Вытягивание в 1D вектор
        x = self.flatten(x)
        
        # Проход через полносвязные слои
        x = self.drop_fc1(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.drop_fc2(F.relu(self.bn_fc2(self.fc2(x))))
        x = self.fc3(x)
        
        return x

def get_model(num_classes, device):
    # Создаем экземпляр нашей кастомной сети
    model = EmotionCNN(num_classes=num_classes)
    
    # Отправляем модель на нужное устройство (CPU, GPU или MPS)
    model = model.to(device)
    
    return model