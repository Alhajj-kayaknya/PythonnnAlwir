import torch
import torchvision
import torchvision.transforms as transforms

from cnncifar10 import MesinVisionCNNcifar10

kelasCIFAR = ('pesawat', 'mobil', 'burung', 'kucing', 'rusa', 'anjing', 'katak', 'kuda', 'kapal', 'truk')

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True)

net = MesinVisionCNNcifar10()

net.load_state_dict(torch.load('./otak_cifar_net.pth'))

net.eval()

dataiter = iter(testloader)
foto, label_asli = next(dataiter)

with torch.no_grad():
    hasil = net(foto)
    _, tebakan = torch.max(hasil, 1)


print("=== HASIL PREDIKSI AI ===")
print("KUNCI JAWABAN ASLI : ", ' | '.join(f'{kelasCIFAR[label_asli[j]]}' for j in range(4)))
print("JAWABAN DARI AI : ", ' | '.join(f'{kelasCIFAR[tebakan[j]]}' for j in range(4)))



