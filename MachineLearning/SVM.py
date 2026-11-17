from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

Xtrain = np.array([
    [2, 150], [3, 160], [4, 155],
    [8, 120], [9, 110], [10, 115]
])

ytrain = np.array([0, 0, 0, 1, 1, 1])

combine = StandardScaler()

xtrainCombine = combine.fit_transform(Xtrain)

MesinSVM = SVC(kernel='linear')
MesinSVM.fit(xtrainCombine, ytrain)

print("Data 2d yang digeprek angka besarnya : ")
print(xtrainCombine)
print("===============================")

print("Index terdepan (support vector) : ")
print(MesinSVM.support_)

print(Xtrain.shape)
print(ytrain.shape)