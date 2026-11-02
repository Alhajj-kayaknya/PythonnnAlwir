import numpy as np

# Matriks kiri (koefisien x, y, z dari soal terakhir lu)
A = np.array([[1, 1, 1],
              [2, 3, -1],
              [3, -1, 2]])

# Matriks kanan (hasil di ujung kanan sama dengan)
B = np.array([2, -1, 13])

# Mesin kuli jalanin Eliminasi Gauss (cuma butuh 0.001 detik)
hasil = np.linalg.solve(A, B)

# Kita buletin angkanya biar kaga ada koma mikroskopis bawaan mesin (kayak 2.9999999)
hasil = np.round(hasil)

print("=== HASIL ELIMINASI GAUSS ===")
print(f"Nilai x = {hasil[0]}")
print(f"Nilai y = {hasil[1]}")
print(f"Nilai z = {hasil[2]}")