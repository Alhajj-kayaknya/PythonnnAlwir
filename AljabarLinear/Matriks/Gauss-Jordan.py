import numpy as np

A = np.array([
    [1, 2, 1],
    [2, -1, 1],
    [1, 1, -2]
])

B = np.array([3, 2, 5])

hasil = np.linalg.solve(A, B)

invers_A = np.linalg.inv(A)

print("=== HASIL X Y Z ===")
print(np.round(hasil))

print("=== INVERS MATRIKS A ===")
print(np.round(invers_A, 2))