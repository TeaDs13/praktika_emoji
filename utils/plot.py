import matplotlib.pyplot as plt

def plot_metrics(train_losses, val_losses, train_acc, val_acc):
    plt.figure()

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="train")
    plt.plot(val_acc, label="val")
    plt.title("Accuracy")
    plt.legend()

    plt.show()