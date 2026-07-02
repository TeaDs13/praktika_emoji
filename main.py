import torch
from torch import nn, optim

from utils.transforms import get_transforms
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
model = get_model(len(train_dataset.classes)).to(device)

# 4. loss + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 5. training loop
best_acc = 0

for epoch in range(10):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)

    print(f"Epoch {epoch+1}")
    print(f"Train: {train_loss:.4f}, {train_acc:.4f}")
    print(f"Val: {val_loss:.4f}, {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "/Users/lomash/programmingPy/practice_2026/models/best_model.pth")
        print("Saved best model")