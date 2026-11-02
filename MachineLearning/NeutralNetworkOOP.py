import numpy as np


class NeuralNetwork :
    def __init__(self) :
        self.wHidden = np.array([[-5.0, 5.0, -10.0, 10.0]])
        self.bHidden = np.array([[1.0, -1.0, 5.0, -8.0]])

        self.wOutput = np.array([
            [2.0],
            [2.0],
            [2.0],
            [2.0]
        ])
        self.bOutput = np.array([[-0.5]])

    def train(self, X, Y, epoch=10000, LR=0.1) :
        print("=== Mesin Mulai Belajar ===")

        for e in range(1, epoch + 1) :
            Z = np.dot(X, self.wHidden) + self.bHidden
            H = np.maximum(0, Z)

            ZOutput = np.dot(H, self.wOutput) + self.bOutput
            pFinal = 1 / (1 + np.exp(-ZOutput))

            errorOutput = pFinal - Y

            H_T = H.T
            GradWOutput = np.dot(H_T, errorOutput)
            GradBOutput = np.sum(errorOutput, axis=0, keepdims=True)

            errorHidden = np.dot(errorOutput, self.wOutput.T)
            TurunanReLu = (H > 0).astype(float)
            errorASELI = errorHidden * TurunanReLu

            X_T = X.T
            gradwHidden = np.dot(X_T, errorASELI)
            gradbHidden = np.sum(errorASELI, axis=0, keepdims=True)

            self.wOutput = self.wOutput - (LR * GradWOutput)
            self.bOutput = self.bOutput - (LR * GradBOutput)

            self.wHidden = self.wHidden - (LR * gradwHidden)
            self.bHidden = self.bHidden - (LR * gradbHidden)

            if e % 10000 == 0 :
                print(f"\n=== Epoch {e} ===")
                print(f"Peluang : \n{np.round(pFinal, 4)}")

    def predict(self, JamBelajar) :
        XBaru = np.array([[JamBelajar]])
        XBaru = XBaru / 20.0

        Z = np.dot(XBaru, self.wHidden) + self.bHidden
        H = np.maximum(0, Z)

        ZOutput = np.dot(H, self.wOutput) + self.bOutput
        pFinal = 1 / (1 + np.exp(-ZOutput))

        persentase = np.round(pFinal[0][0] * 100, 2)

        print(f"\n--- HASIL PREDIKSI ---")
        print(f"Jam Belajar   : {JamBelajar} jam")
        print(f"Peluang Lulus : {persentase}%")


X = np.array([[8.0], [5.0], [2.0], [0.5], [15.0], [0.0], [20.0]])
Y = np.array([[1.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0]])

X = X / 20.0

Mesin_Gw = NeuralNetwork()
Mesin_Gw.train(X, Y, epoch=100000, LR=0.01)

Mesin_Gw.predict(3.0)
Mesin_Gw.predict(10.0)
Mesin_Gw.predict(25.0)


# ketika ai gagal/kebingungan dia bakal ngasih tebakan 50/50
# kondisi ini disebut sebagai linear separability capacity
