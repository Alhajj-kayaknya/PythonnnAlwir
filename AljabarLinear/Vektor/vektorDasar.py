import numpy as np

vektor_a = np.array([5,8])
vektor_b = np.array([2,3])
vektor_c = np.array([1,1])
vektor_d = np.array([9,10])

HasilTambah = vektor_a + vektor_b
HasilKurang = vektor_a - vektor_b
HasilTambahKurang = vektor_a + vektor_b - vektor_c
HasilTambah4Vektor = vektor_a + vektor_b + vektor_c + vektor_d
HasilKurang4Vektor = vektor_a - vektor_b - vektor_c - vektor_d

print(f"Hasil penjumlahan a+b : {HasilTambah}")
print(f"Hasil pengurangan a-b : {HasilKurang}")
print(f"\nHasil a+b-c : {HasilTambahKurang}")
print(f"\nHasil tambah 4 Vektor : {HasilTambah4Vektor}")
print(f"Hasil kurang 4 Vektor : {HasilKurang4Vektor}")