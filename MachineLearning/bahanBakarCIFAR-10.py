import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Proses")
dataLatihan = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

uploadLatihan = torch.utils.data.DataLoader(dataLatihan, batch_size=4, shuffle=True)
print("Proses Selesai")