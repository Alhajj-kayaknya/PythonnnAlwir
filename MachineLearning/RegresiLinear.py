import numpy as np


xData = np.array([2.0, 4.0, 0.0])
yTarget = np.array([30.0, 40.0, 20.0])

w = 0.0
b = 0.0
learning_rate = 0.13 #lr 0.01 = 952, lr 0.02 = 475, lr 0.03 = 316, lr 0.04 = 236, lr 0.05 = 189, lr 0.0
jumlahEpoch = 2000

print("=== Start Training Regresi Linear ===")

for i in range(jumlahEpoch):
    y_tebakan = (w * xData) + b # rumus asli

    errorSekarang = np.mean((y_tebakan - yTarget) ** 2 ) # fungsi error

    grad_W = np.mean(2 * (y_tebakan - yTarget) * xData) # fungsi turunan buat w
    grad_B = np.mean(2 * (y_tebakan - yTarget)) # fungsi turunan buat b

    w = w - (learning_rate * grad_W) # update w
    b = b - (learning_rate * grad_B) # update b

    if errorSekarang < 0.0001 : # rem tangan
        print(f"\n [!] Early Stopping Terpicu Di Epoch : {i+1} ")
        break

    if (i+1) % 100 == 0 : # tampilkan epoch kelipatan 100
        print(f"Epoch {i+1} | w : {w:.4f} | b : {b:.4f} | sisa error : {errorSekarang:.4f}")

print("\n=== Training Kelar ===")
print(f"Hasil Akhir Otak AI -> kemiringan (w) : {w:.4f}, bias (b) : {b:.4f}")