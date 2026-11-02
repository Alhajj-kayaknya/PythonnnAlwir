import numpy as np

skalar_k = 4
vektor_u = np.array([3, -1])

bobot_ai = np.array([0.8, -0.5])
error_ai = np.array([2.0, -1.0])
learning_rate = 0.1

bobot = np.array([0.5, 0.2])
error = np.array([1.0, 2.0])
rate = 0.01

hasil_1 = bobot - error * rate
print(hasil_1)
hasil = bobot_ai - error_ai * learning_rate
print(hasil)

hasil_kali = skalar_k * vektor_u
print(f"\nhasil perkalian skalar dengan vektor u adalah : {hasil_kali}")