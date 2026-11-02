import numpy as np

X = np.array([
    [8.0],
    [5.0],
    [2.0]
])

wHidden = np.array([[0.5, -0.2]])
bHidden = np.array([[0.1, 0.0]])

wOutput = np.array([
    [0.8],
    [0.4]
])

bOutput = np.array([[-0.5]])

Y = np.array([
    [1.0],
    [1.0],
    [0.0]
])
for epoch in range(1, 2001) :
    Z = np.dot(X, wHidden) + bHidden
    H = 1 / (1 + np.exp(-Z))

    ZOutput = np.dot(H, wOutput) + bOutput

    pFinal = 1 / (1 + np.exp(-ZOutput))

    errorOutput = pFinal - Y

    H_T = H.T

    gradWOutput = np.dot(H_T, errorOutput)

    gradBOutput = np.sum(errorOutput, axis=0, keepdims=True)

    errorHidden = np.dot(errorOutput, wOutput.T)

    TurunanSigmoid = H * (1.0 - H)

    errorASELI = errorHidden * TurunanSigmoid

    X_T = X.T

    gradWhidden = np.dot(X_T, errorASELI)
    gradBhidden = np.sum(errorASELI, axis=0, keepdims=True)

    LR = 0.1

    wOutput = wOutput - (LR * gradWOutput)
    bOutput = bOutput - (LR * gradBOutput)

    wHidden = wHidden - (LR * gradWhidden)
    bHidden = bHidden - (LR * gradBhidden)

    if epoch % 1000 == 0 :
        print(f"\nEpoch : {epoch}")
        print(f"Peluang Lulus final : {np.round(pFinal, 4)}")
        print(f"output error : {np.round(errorASELI, 4)}")
        print(f"grad w : {np.round(gradWhidden, 4)}")
        print(f"grad b : {np.round(gradBhidden, 4)}")
        print(f"W baru : {np.round(wHidden, 4)}")
        print(f"B baru : {np.round(bHidden, 4)}")


print("\n" + "="* 30)
print("Menguji Machine Learning")
print("="* 30)

XTest = np.array([
    [7.0],
    [3.0],
    [1.0],
    [0.5]
])

ZTest = np.dot(XTest, wHidden) + bHidden
HTest = 1 / (1 + np.exp(-ZTest))

ZOutputTest = np.dot(HTest, wOutput) + bOutput
pFinalTest = 1 / (1 + np.exp(-ZOutputTest))

print("=== PERSENTASE KELULUSAN ===")
print(f"Peluang Anak ke-4 : {np.round(pFinalTest[0][0] * 100, 2)}%")
print(f"Peluang Anak ke-5 : {np.round(pFinalTest[1][0] * 100, 2)}%")
print(f"Peluang Anak ke-6 : {np.round(pFinalTest[2][0] * 100, 2)}%")
print(f"Peluang Anak ke-7 : {np.round(pFinalTest[3][0] * 100, 2)}%")


