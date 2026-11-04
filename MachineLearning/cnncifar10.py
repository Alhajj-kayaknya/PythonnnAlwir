import torch
import torch.nn as nn
import torch.nn.functional as F

class MesinVisionCNNcifar10(nn.Module):
    def __init__(self):
        super(MesinVisionCNNcifar10, self).__init__()

        # lapisan 1 buat RGB
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)

        # lapisan 2 buat zoom agar makin detail
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)

        # dibagi 2 resolusi gambarnya
        self.pool = nn.MaxPool2d(2, 2)

        # dijadiin matrix 1d yang asalnya 3d
        self.flatten = nn.Flatten()

        # 32 = tebelnya, 8 = tingginya, 8 = lebarnya
        self.fc1 = nn.Linear(32 * 8 * 8, 128)

        # outputnya 10 karna cifar cuma ada 10 kategori
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # disapu sama lapisan 1 -> aktivasi -> dipress
        x = self.pool(F.relu(self.conv1(x)))

        # disapu sama lapisan 2 -> aktivasi -> dipress
        x = self.pool(F.relu(self.conv2(x)))

        # digepengin jadi matriks 1d
        x = self.flatten(x)

        # masuk ke output akhir
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

RGB = MesinVisionCNNcifar10()
print(RGB)











