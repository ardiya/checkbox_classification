import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.n_classes = n_classes
        self.pool = nn.MaxPool2d(2, 2)
        self.conv0 = nn.Conv2d(3, 16, 5, padding="same")
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(16, 32, 3, padding="same")
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, 1, padding="same")
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(32, 64, 3, padding="same")
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 1, padding="same")
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(64, 128, 3, padding="same")
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(128, 128, 1, padding="same")
        self.relu6 = nn.ReLU()

        self.dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(128, 128)
        self.reluf1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 16)
        self.reluf2 = nn.ReLU()
        self.fc3 = nn.Linear(16, self.n_classes)

    def forward(self, x):
        x = self.relu0(self.conv0(x))
        x = self.pool(x)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool(x)
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.pool(x)

        x = torch.flatten(F.adaptive_max_pool2d(x, output_size=1), start_dim=1)
        x = self.dropout(x)
        x = self.reluf1(self.fc1(x))
        x = self.fc2(x)
        embedding = x

        classification = self.fc3(self.reluf2(embedding))

        return embedding, classification

    @staticmethod
    def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))
            nn.init.constant_(m.bias.data, 0)
