import numpy as np
from numpy import linalg

vektor_a = np.array([6,8])
vektor_b = np.array([2,3,5])
vektor_c = np.array([3,4,12])
vektor_d = np.array([4,5,6])
vektor_e = np.array([10,11,13])
#ngitung panjang vektor pake linalg(linear algebra)
panjang_a = linalg.norm(vektor_a)
print(f"Panjang vektor a adalah : {panjang_a}")

panjang_b = linalg.norm(vektor_b)
print(f"Panjang vektor 3D ini adalah : {panjang_b}")

panjang_c = linalg.norm(vektor_c)
print(f"Panjang vektor 3D ini adalah : {panjang_c}")

panjang_d = linalg.norm(vektor_d)
print(f"\nPanjang vektor 3D ini adalah : {panjang_d}")

panjang_e = linalg.norm(vektor_e)
print(f"Panjang vektor 3D ini adalah : {panjang_e}")

panjang_dua_vektor = panjang_d + panjang_e
print(f"Panjang dua Vektor adalah : {panjang_dua_vektor}")