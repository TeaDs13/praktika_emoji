import numpy as np
from torchvision import datasets

class AlbumentationsWrapper:
    """Обертка, которая заставляет ImageFolder дружить с Albumentations"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        # Конвертируем PIL Image (который читает ImageFolder) в массив numpy
        img_np = np.array(img)
        # Применяем аугментацию Albumentations
        augmented = self.transform(image=img_np)
        # Возвращаем тензор
        return augmented['image']


def load_datasets(train_dir, test_dir, train_tf, test_tf):
    # Оборачиваем наши трансформации в обертку перед передачей в ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=AlbumentationsWrapper(train_tf))
    test_dataset = datasets.ImageFolder(test_dir, transform=AlbumentationsWrapper(test_tf))
    
    return train_dataset, test_dataset