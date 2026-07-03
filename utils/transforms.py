from torchvision import transforms

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        # ПРИНУДИТЕЛЬНО делаем картинку 1-канальной (черно-белой)
        transforms.Grayscale(num_output_channels=1), 
        
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        # И здесь тоже принудительно 1 канал
        transforms.Grayscale(num_output_channels=1), 
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    return train_transform, test_transform