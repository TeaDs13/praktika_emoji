import matplotlib.pyplot as plt


class Logger:
    def __init__(self):
        # метрики
        self.train_loss = []
        self.train_acc = []

        self.val_loss = []
        self.val_acc = []
        self.val_f1 = []

        self.lr = []

    # ---------------------------
    # запись метрик за эпоху
    # ---------------------------
    def log(self, train_loss, train_acc,
            val_loss, val_acc, val_f1,
            lr):

        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)

        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        self.val_f1.append(val_f1)

        self.lr.append(lr)

    # ---------------------------
    # LOSS график
    # ---------------------------
    def plot_loss(self):
        plt.figure()
        plt.plot(self.train_loss, label="Train Loss")
        plt.plot(self.val_loss, label="Val Loss")
        plt.title("Loss over epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.show()

    # ---------------------------
    # ACCURACY график
    # ---------------------------
    def plot_accuracy(self):
        plt.figure()
        plt.plot(self.train_acc, label="Train Acc")
        plt.plot(self.val_acc, label="Val Acc")
        plt.title("Accuracy over epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid()
        plt.show()

    # ---------------------------
    # F1 график (самый важный)
    # ---------------------------
    def plot_f1(self):
        plt.figure()
        plt.plot(self.val_f1, label="Val F1 (macro)", color="red")
        plt.title("F1-score over epochs")
        plt.xlabel("Epoch")
        plt.ylabel("F1-score")
        plt.legend()
        plt.grid()
        plt.show()

    # ---------------------------
    # Learning rate график
    # ---------------------------
    def plot_lr(self):
        plt.figure()
        plt.plot(self.lr, label="Learning Rate")
        plt.title("Learning Rate schedule")
        plt.xlabel("Epoch")
        plt.ylabel("LR")
        plt.yscale("log")
        plt.legend()
        plt.grid()
        plt.show()

    # ---------------------------
    # ВСЁ СРАЗУ
    # ---------------------------
    def plot_all(self):
        self.plot_loss()
        self.plot_accuracy()
        self.plot_f1()
        self.plot_lr()