import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def get_transforms():
    train_transform = A.Compose([
        A.Resize(48, 48),  # Или размер, который требует твоя модель (например, 224)
        
        # 1. Отражение по горизонтали (эмоция не меняется от смены стороны)
        A.HorizontalFlip(p=0.5),
        
        # 2. Небольшие повороты и сдвиги (симулирует наклон головы)
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=0),
        
        # 3. Случайное изменение яркости и контраста (лица в датасете освещены по-разному)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        
        # 4. Искажения (GridDistortion) — помогает модели быть устойчивой к разным формам лиц
        A.OneOf([
            A.GridDistortion(p=1.0),
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=1.0),
        ], p=0.3),
        
        # 5. Нормализация
        A.Normalize(mean=(0.485,), std=(0.229,)), # Настрой под каналы (1 или 3)
        ToTensorV2(),
    ])

    test_transform = A.Compose([
        A.Resize(48, 48),
        A.Normalize(mean=(0.485,), std=(0.229,)),
        ToTensorV2(),
    ])
    
    # Чтобы использовать albumentations в PyTorch Dataset, 
    # в самом Dataset класс transform нужно вызывать как: transform(image=img)['image']
    return train_transform, test_transform