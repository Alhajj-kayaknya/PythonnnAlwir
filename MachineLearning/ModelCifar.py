import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from cnncifar10 import MesinVisionCNNcifar10

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

net = MesinVisionCNNcifar10()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print("=== Mulai Proses Latihan Model CIFAR10 ===")

for epoch in range(2) :
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0) :
        # pisahin data input gambar sama jawaban asli
        inputs, labels = data

        # bersihin gradient dari loop sebelumnya
        optimizer.zero_grad()

        # forward pass
        outputs = net(inputs)

        # hitung loss (selisih output model dan jawaban asli)
        loss = criterion(outputs, labels)

        # backpropagation
        loss.backward()

        #update bobot/weight
        optimizer.step()

        # print proses per 2000
        running_loss += loss.item()
        if i % 2000 == 1999 :
            print(f"[Epoch {epoch+1}, Batch {i+1}] loss : {running_loss / 2000:.3f}")
            running_loss = 0.0

print("=== Training Ai Selesai ===")
PATH = './otak_cifar_net.pth'
torch.save(net.state_dict(), PATH)

