import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

class MesinVisionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3136, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

AI_Baru = MesinVisionCNN()

AI_Baru.load_state_dict(torch.load('otak_cnn_97persen.pth', weights_only=True))
AI_Baru.eval()

print("--- OTAK 97% BERHASIL DILOAD, SIAP NEBAK GAMBAR ---\n")

transform_ujian = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

try :
    gambarAsli = Image.open('angkaku.png')
    gambarTensor = transform_ujian(gambarAsli)
    gambarTensor = gambarTensor.unsqueeze(0)

    with torch.no_grad():
        tebakanMentah = AI_Baru(gambarTensor)
        _, tebakanFinal = torch.max(tebakanMentah, 1)

    print("=========================================")
    print(f"Hasil Tebakan Ai : {tebakanFinal.item()} !!!")
    print("=========================================")

except FileNotFoundError :
    print("Tidak Ada Input Foto, Mohon input foto terlebih dahulu ")






