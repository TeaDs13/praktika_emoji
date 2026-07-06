import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# CHANNEL ATTENTION
# =========================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        avg = self.avg_pool(x).view(b, c)
        max = self.max_pool(x).view(b, c)

        avg_out = self.mlp(avg)
        max_out = self.mlp(max)

        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1, 1)

        return x * out


# =========================
# SPATIAL ATTENTION
# =========================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(
            2, 1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x_cat = torch.cat([avg_out, max_out], dim=1)

        out = self.conv(x_cat)
        out = self.sigmoid(out)

        return x * out


# =========================
# CBAM BLOCK
# =========================
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# =========================
# EMOTION CNN + CBAM
# =========================
class EmotionCNN_CBAM(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN_CBAM, self).__init__()

        # -------------------------
        # Block 1
        # -------------------------
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.cbam1 = CBAM(64)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d(0.25)

        # -------------------------
        # Block 2
        # -------------------------
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.cbam2 = CBAM(128)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout2d(0.25)

        # -------------------------
        # Block 3
        # -------------------------
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.cbam3 = CBAM(256)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout2d(0.25)

        # -------------------------
        # Block 4
        # -------------------------
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.cbam4 = CBAM(512)
        self.pool4 = nn.MaxPool2d(2)
        self.drop4 = nn.Dropout2d(0.25)

        # -------------------------
        # CLASSIFIER
        # -------------------------
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(512 * 3 * 3, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.drop_fc1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.drop_fc2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):

        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.cbam1(x)
        x = self.pool1(x)
        x = self.drop1(x)
# Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.cbam2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.cbam3(x)
        x = self.pool3(x)
        x = self.drop3(x)

        # Block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.cbam4(x)
        x = self.pool4(x)
        x = self.drop4(x)

        # Classifier
        x = self.flatten(x)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.drop_fc1(x)

        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.drop_fc2(x)

        x = self.fc3(x)

        return x


# =========================
# GET MODEL FUNCTION
# =========================
def get_model2(num_classes, device):
    model = EmotionCNN_CBAM(num_classes=num_classes)
    return model.to(device)