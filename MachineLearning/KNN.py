from sklearn.neighbors import KNeighborsClassifier
import numpy as np

X_train = np.array([[2,1], [3,2], [6,6], [7,6]])

# 0 = senjata, 1 = ramuan
Y_train = np.array([0,0,1,1])

MesinKNN = KNeighborsClassifier(n_neighbors=3)

MesinKNN.fit(X_train, Y_train)

# uji tes titik (4,3)

zMisterius = np.array([[4,3]])
tebakanZ = MesinKNN.predict(zMisterius)

if tebakanZ[0] == 0 :
    print("Mesin nebak : Senjata")
else :
    print("Mesin nebak : Ramuan")




