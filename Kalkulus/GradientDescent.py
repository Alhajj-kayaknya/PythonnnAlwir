import numpy as np

def fungsi_error(x) :
    return 5 * (x ** 2)

def fungsi_turunan(x) :
    return 10 * x

posisi_x = 3.0
learning_rate = 0.05
JumlahEpoch = 100

print("Start Training AI")
print(f"Tebakan Awal {posisi_x} | Error awal = {fungsi_error(posisi_x)}\n")

for i in range(JumlahEpoch) :
    # cari kompas kemiringan
    gradient = fungsi_turunan(posisi_x)

    # update posisi tebing
    posisi_x = posisi_x - learning_rate * gradient

    # cek sisa error sekarang
    errorSekarang = fungsi_error(posisi_x)

    # print hasil biar keliatan perbedaanya
    print(f"Epoch {i+1} | Posisi X : {posisi_x:.4f} | Sisa Error : {errorSekarang:.4f}")

    if errorSekarang < 0.0001 :
        break



print("\n=== Training Selesai ===")