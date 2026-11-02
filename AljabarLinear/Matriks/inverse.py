import numpy as np
from numpy import linalg

matriks_K = np.array([[-3, 4],
                      [-2, 2]])
matriks_Z = np.array([[10, 20],
                      [4, 5]])

det_K = linalg.det(matriks_K)
inverse_K = linalg.inv(matriks_K)

det_Z = linalg.det(matriks_Z)
inverse_Z = linalg.inv(matriks_Z)

print(f"Hasil Determinan K = {det_K}")
print(f"Hasil Inverse K = {inverse_K}")

print(f"\nHasil Determinan Z = {det_Z}")
print(f"Hasil Inverse Z = {inverse_Z}")