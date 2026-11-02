import numpy as np

matriks_X = np.array([[2,-1], [0,4]])
matriks_Y = np.array([[3,5], [2,-2]])

hasilJumlah = matriks_X + matriks_Y
hasilAkhir = hasilJumlah.T

print(f"Hasil Tambah = {hasilJumlah}")
print(f"Hasil Transpose = {hasilAkhir}")