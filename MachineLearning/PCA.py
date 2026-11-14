from sklearn.decomposition import PCA
import numpy as np

XTrain = np.array([
    [5000, 10, 0],
    [5200, 11, 1],
    [4000, 7, 0],
    [4100, 8, 1]
])

print(f"Bentuk data awal : {XTrain.shape}")

mesinPCA = PCA(n_components=2)

mesinPCA.fit(XTrain)

XPress = mesinPCA.transform(XTrain)

print(f"Bentuk data setelah diPress : {XPress.shape}")
print("\nIsi data setelah dikompres : ")
print(XPress)

