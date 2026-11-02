import numpy as np

x = np.array([8.0])
y = np.array([1.0])

wHidden = np.array([0.5,-0.2]) #w1 w2
bHidden = np.array([0.1,0.0]) #b1 b2

wOutput = np.array([0.8,0.4]) #w3 dan w4
bOutput = np.array([-0.5]) #b3

for epoch in range(1, 2001) :
    z = (wHidden * x) + bHidden
    h = 1 / (1 + np.exp(-z))

    zOutput = np.sum(wOutput * h) + bOutput
    pFinal = 1 / (1 + np.exp(-zOutput))

    # GRADIENT DESCENT
    # 1. turunan parsial dari loss function terhadap output
    error_out = pFinal - y

    # 2. ngitung turunan error terhadap hidden layer
    grad_h1 = error_out * wOutput[0]

    # 3. kurva sigmoid h1 dan nabrak input x buat dapet turunan w1
    grad_w1 = grad_h1 * (h[0] * (1 - h[0])) * x[0]

    LR = 0.1

    wLama = wHidden[0]
    wHidden[0] = wHidden[0] - (LR * grad_w1[0])


    if epoch % 500 == 0 :
        print(f"\nEpoch ke : {epoch}")
        print(f"Peluang Lulus final : {np.round(pFinal, 4)}")
        print(f"output error : {np.round(error_out, 4)}")
        print(f"grad_h1 : {np.round(grad_h1, 4)}")
        print(f"grad_w1 : {np.round(grad_w1, 4)}")

        print("=== Hasil Cambuk ai ===")
        print(f"w1 Lama : {np.round(wLama, 4)} | w1 Baru : {np.round(wHidden[0], 4)}")

print("=== HASIL FINAL ===")
print(f"w1 Lama : {np.round(wLama, 4)} | w1 Baru : {np.round(wHidden[0], 4)}")


# kasih kesempatan untuk hamba untuk terus berkembang dan mengangkat derajat keluarga