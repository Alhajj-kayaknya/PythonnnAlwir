from sklearn.ensemble import RandomForestClassifier
import numpy as np

XTrain = np.array([
    [40,20],
    [99,300],
    [35,15],
    [95,280]
])

YTrain = np.array([0, 1, 0, 1])

MesinRF = RandomForestClassifier(n_estimators=5, random_state=42)

MesinRF.fit(XTrain, YTrain)

playermisterius = np.array([[65,250]])

tebakan = MesinRF.predict(playermisterius)

if tebakan[0] == 1 :
    print("Hasil Voting RF : CHEATER")
else :
    print("Hasil Voting RF : BERSIH")





