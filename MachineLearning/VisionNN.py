import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random


transform = transforms.Compose([transforms.ToTensor()])
dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

class AImata(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.flatten = nn.Flatten()

        self.hidden = nn.Linear(784, 128)
        self.ReLU = nn.ReLU()

        self.output = nn.Linear(128, 10)

    def forward(self, x) :
        x = self.flatten(x)
        x = self.ReLU(self.hidden(x))
        x = self.output(x)
        return x


MesinVision = AImata()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(MesinVision.parameters(), lr=0.001)

print("--- MULAI DOWNLOAD DAN TRAINING MATA AI ---")
print("--- SEDANG PROSES ---")

for batch_idx, (gambar, label) in enumerate(dataloader) :
    optimizer.zero_grad()
    prediksi = MesinVision(gambar)
    loss = loss_function(prediksi, label)
    loss.backward()
    optimizer.step()

    if batch_idx % 300 == 0 :
        print(f"Batch : {batch_idx} Selesai, Loss : {loss.item():.4f}")

print("--- TRAINING SELESAI ---")

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

index_random = random.randint(0, 9999)
gambar_tes, kunci_tes = test_dataset[index_random]

MesinVision.eval()
with torch.no_grad() :
    tebakan_mentah = MesinVision(gambar_tes)
    tebakan_final = torch.argmax(tebakan_mentah).item()


print("--------------")
print(f"JAWABAN ASLI : {kunci_tes}")
print(f"TEBAKAN AI : {tebakan_final}")

plt.imshow(gambar_tes.squeeze(), cmap='gray')
plt.title(f"TEBAKAN AI : {tebakan_final} | TEBAKAN ASLI : {kunci_tes}")
plt.show()

# UJIAN BUAT MACHINE LEARNING

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

MesinVision.eval()

totalBenar = 0
totalSoal = 0

print("\n--- MULAI TES ---")

with torch.no_grad() :
    for gambar, labelAsli in test_dataloader :
        tebakan_mentah = MesinVision(gambar)

        _, tebakan_final = torch.max(tebakan_mentah, 1)

        totalBenar += (tebakan_final == labelAsli).sum().item()
        totalSoal += labelAsli.size(0)

akurasi = (totalBenar / totalSoal) * 100
print(f"Total Soal Ujian : {totalSoal} gambar")
print(f"Tebakan Benar : {totalBenar} gambar")
print(f"Akurasi AI: {akurasi} %")















