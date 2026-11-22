from sklearn.cluster import KMeans
import numpy as np

Xtrain = np.array([
    [1,2], [3,4],
    [7,8], [9,10]
])

MesinKmeans = KMeans(n_clusters=2, random_state=42)

MesinKmeans.fit(Xtrain)

print(Xtrain.shape)
print("Hasil pembagian kelompok (labels) : ")
print(MesinKmeans.labels_)

print("Centroid : ")
print(MesinKmeans.cluster_centers_)

databaru = np.array([[8,9]])
print("Data masuk Kelompok : ", MesinKmeans.predict(databaru))



