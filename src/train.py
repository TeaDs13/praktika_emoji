def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device).float(), y.to(device).long()

        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, pred = outputs.max(1)
        total += y.size(0)
        correct += (pred == y).sum().item()

    return total_loss / len(loader), correct / total