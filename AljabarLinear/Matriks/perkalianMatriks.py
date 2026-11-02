import numpy as np

matriks_A = np.array([[1, 2],
                      [3, 4]])
matriks_B = np.array([[5, 6],
                      [7, 8]])

matriksBobot = np.array([[0.5, 0.2],
                         [0.1, 0.8]])
matriksData = np.array([[8, 4],
                        [2,6]])

matriks_M = np.array([[2, 1],
                      [1, 2]])
matriks_N = np.array([[3, 1],])

HasilKali = matriks_A @ matriks_B
print(f"Hasil kali = {HasilKali}")

hasilPrediksiCuaca = matriksBobot @ matriksData
print(f"\nHasil Prediksi{hasilPrediksiCuaca}")

matriks_N_Baru = matriks_N.T
hasilKali_denganManipulasiTranspose = matriks_M @ matriks_N_Baru
print(f"\nHasil kali Dengan manipulasi : {hasilKali_denganManipulasiTranspose}")