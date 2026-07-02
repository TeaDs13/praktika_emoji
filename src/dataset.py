from torchvision import datasets

def load_datasets(train_dir, test_dir, train_tf, test_tf):
    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_tf)

    return train_dataset, test_dataset