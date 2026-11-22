from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

X_Train = np.array([
    [10, 2],
    [20, 5],
    [30, 8],
    [40, 12],
    [50, 15]
])

Y_Train = np.array([30, 50, 70, 90, 110])

MesinGradient = GradientBoostingRegressor(n_estimators=50, learning_rate=0.2, random_state=42)
MesinGradient.fit(X_Train, Y_Train)

akunBaru = np.array([[36, 11]])
tebakan = MesinGradient.predict(akunBaru)

print(f"Tebakan Harga Akun : Rp. {tebakan[0]:.2f} ribu")