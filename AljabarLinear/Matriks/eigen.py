import numpy as np
from numpy import linalg

B = np.array([
    [2, 1],
    [1, 2]
])

EigenValue, EigenVector = linalg.eig(B)

print("=== HASIL EIGEN MATRIKS B ===")
print(f"EigenValue (lambda) : {EigenValue}")
print(f"EigenVector (panah) : {EigenVector}")

panahKiri = EigenVector[:, 0]
lambda_3 = EigenValue[0]

sisiKiri = np.dot(B, panahKiri)
sisiKanan = lambda_3 * panahKiri

print("\n=== PEMBUKTIAN ===")
print(f"B  * v : {sisiKiri}")
print(f"lambda * v : {sisiKanan}")