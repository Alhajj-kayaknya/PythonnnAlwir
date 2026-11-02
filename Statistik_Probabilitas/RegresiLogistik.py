import numpy as np

def sigmoid(z) :
    return 1/(1+np.exp(-z))

w = 2
b = -5
x = 1

z_murid_B = (w * x) + b
peluang = sigmoid(z_murid_B)

a = 2
b = -5
c = 7

z_murid_A = (a * c) + b
peluangTes = sigmoid(z_murid_A)

print(f"Peluang Lulus : {peluang * 100:.2f}%")
print(f"\nPeluang Tes: {peluangTes * 100:.2f}%")

