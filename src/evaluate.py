from sklearn.metrics import f1_score
import torch

def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item() * x.size(0)

            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    f1 = f1_score(all_targets, all_preds, average="macro")

    return total_loss / total, acc, f1