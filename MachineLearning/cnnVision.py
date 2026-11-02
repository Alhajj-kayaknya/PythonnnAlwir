import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cross_entropy

transforms = transforms.Compose([transforms.ToTensor()])

trainDataSet = torchvision.datasets.MNIST(root='../', train=True, transform=transforms, download=True)
trainDataLoader = torch.utils.data.DataLoader(trainDataSet, batch_size=64, shuffle=True)

testDataSet = torchvision.datasets.MNIST(root='../', train=False, transform=transforms, download=True)
testDataLoader = torch.utils.data.DataLoader(testDataSet, batch_size=64, shuffle=False)


class MesinVisionCNN(nn.Module):
    def __init__(self) :
        super().__init__()
        # conv1 = Kaca Pembesar (Mendeteksi pattern/bentuk)
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)

        # pool = Tukang Kompres (Ngecilin gambar 28x28 jadi 14x14)
        self.pool = nn.MaxPool2d(kernel_size=2)

        # flatten = Mesin Gepengin (Ngeubah matriks 2D jadi array 1D)
        self.flatten = nn.Flatten()

        # fc1 (Fully Connected) = Otak Penebak Akhir (3136 -> 10 tebakan angka)
        self.fc1 = nn.Linear(3136, 10)

    def forward(self, x) :
        x = self.conv1(x) # Gambar discan
        x = torch.relu(x) # aktivasi biar ga kaku
        x = self.pool(x) # gambar dikompres
        x = self.flatten(x) # diubah jadi matriks 1d
        x = self.fc1(x) # nebak hasil akhirnya
        return x

MesinVision = MesinVisionCNN()
loss_fn = CrossEntropyLoss()
optimizer = optim.Adam(MesinVision.parameters(), lr=0.001)

print("--- MULAI TRAINING CNN ---")
epochs = 3
for epoch in range(epochs) :
    MesinVision.train()
    totalLoss = 0
    for gambar, labelAsli in trainDataLoader :
        optimizer.zero_grad()
        tebakan = MesinVision(gambar)
        loss = loss_fn(tebakan, labelAsli)
        loss.backward()
        optimizer.step()
        totalLoss += loss.item()
    print(f"epoch {epoch+1}/{epochs}  loss: {totalLoss / len(trainDataLoader)}")


print("--- MULAI TES MACHINE ---")
MesinVision.eval()
totalBenar = 0
totalSoal = 0

with torch.no_grad() :
    for gambar, labelAsli in testDataLoader :
        tebakanMentah = MesinVision(gambar)
        _, tebakanFinal = torch.max(tebakanMentah, 1)
        totalBenar += (tebakanFinal == labelAsli).sum().item()
        totalSoal += labelAsli.size(0)

akurasi = (totalBenar / totalSoal) * 100
print(f"\nTotal Soal : {totalSoal} Gambar")
print(f"Total Benar : {totalBenar} Gambar")
print(f"Akurasi AI : {akurasi} %")

# 6. SAVE GAME OTAK AI
torch.save(MesinVision.state_dict(), 'otak_cnn_97persen.pth')
print("\n[INFO] Otak AI berhasil disave ke 'otak_cnn_97persen.pth'!")












