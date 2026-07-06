import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# CBAM
# =========================

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        avg = self.mlp(self.avg_pool(x).view(b, c))
        max_ = self.mlp(self.max_pool(x).view(b, c))

        out = avg + max_
        return self.sigmoid(out).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)

        x = torch.cat([avg, max_], dim=1)
        x = self.conv(x)

        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# =========================
# Residual + CBAM Block
# =========================

class ResidualCBAMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.cbam = CBAM(out_channels)

        # shortcut (skip connection)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        x = self.cbam(x)

        x = x + residual
        return F.relu(x)


# =========================
# FULL MODEL
# =========================

class EmotionResCBAM(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Residual stages
        self.layer1 = ResidualCBAMBlock(64, 64)
        self.layer2 = ResidualCBAMBlock(64, 128, stride=2)
        self.layer3 = ResidualCBAMBlock(128, 256, stride=2)

        # Pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(p=0.4)

        # Classifier
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        return self.fc(x)


# =========================
# GET MODEL
# =========================

def get_model(num_classes, device):
    model = EmotionResCBAM(num_classes)
    return model.to(device)