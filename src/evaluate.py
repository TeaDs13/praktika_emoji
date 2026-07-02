import torch


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item()

            _, pred = outputs.max(1)
            total += y.size(0)
            correct += (pred == y).sum().item()

    return total_loss / len(loader), correct / total