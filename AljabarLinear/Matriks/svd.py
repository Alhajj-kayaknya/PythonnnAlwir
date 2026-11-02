import numpy as np
import matplotlib.pyplot as plt

gambar = plt.imread("images.jpg")

if len(gambar.shape) == 3 :
    gambar_gray = np.mean(gambar,axis=2)
else :
    gambar_gray = gambar

print(f"Ukuran asli foto matriks : {gambar_gray.shape}")

U, Sigma, VT = np.linalg.svd(gambar_gray, full_matrices=False)

print(f"Jumlah Singular Value : {len(Sigma)}")

k = 200
SigmaK = np.diag(Sigma[:k])
UK = U[:, :k]
VT = VT[:k, :]

gambarKompres = np.dot(UK, np.dot(SigmaK, VT))

plt.Figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.title("Foto asli (matriks utuh)")
plt.imshow(gambar_gray, cmap="gray")

plt.subplot(1, 2, 2)
plt.title(f"Hasil SVD (pakai {k} singular value)")
plt.imshow(gambarKompres, cmap="gray")

plt.show()







