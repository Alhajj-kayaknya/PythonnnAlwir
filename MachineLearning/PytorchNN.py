import torch
import torch.nn as nn

X = torch.tensor([[0.0], [5.0], [8.0], [10.0], [12.0], [15.0], [20.0]]) / 20.0
Y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0]])

class MesinPyTorch(nn.Module):
    def __init__(self) :
        super().__init__()
        self.hidden_layer = nn.Linear(1, 20)
        self.ReLU = nn.ReLU()
        self.output_layer = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x) :
        x = self.ReLU(self.hidden_layer(x))
        x = self.sigmoid(self.output_layer(x))
        return x

mesinGw = MesinPyTorch()

lossFunction = nn.MSELoss()
optimizer = torch.optim.Adam(mesinGw.parameters(), lr=0.05)

print("--- MULAI TRAINING PYTHON ---")
for epoch in range(10000):
    optimizer.zero_grad()
    prediksi = mesinGw(X)
    loss = lossFunction(prediksi, Y)

    loss.backward()
    optimizer.step()

    if epoch % 2000 == 0 :
        print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")


anakBaru = torch.tensor([[10.0]]) / 20.0 # lulus
anakBaru2 = torch.tensor([[20.0]]) / 20.0 # gagal
anakBaru3 = torch.tensor([[25.0]]) / 20.0 # gagal
anakBaru4 = torch.tensor([[0.0]]) / 20.0 # lulus
anakBaru5 = torch.tensor([[9.5]]) / 20.0
hasil = mesinGw(anakBaru)
hasil2 = mesinGw(anakBaru2)
hasil3 = mesinGw(anakBaru3)
hasil4 = mesinGw(anakBaru4)
hasil5 = mesinGw(anakBaru5)
print(f"Prediksi : {hasil.item() * 100:.2f}%")
print(f"Prediksi : {hasil2.item() * 100:.2f}%")
print(f"Prediksi : {hasil3.item() * 100:.2f}%")
print(f"Prediksi : {hasil4.item() * 100:.2f}%")
print(f"Prediksi : {hasil5.item() * 100:.2f}%")



