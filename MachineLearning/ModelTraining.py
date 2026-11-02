import numpy as np

x = 10
y = 0

w = 0.0
b = 0.0
learning_rate = 0.1

print("=== Mulai Training AI ===")

for epoch in range (1,6) :
    z = (w * x) + b

    p = 1 / (1 + np.exp(-z))
    error = -(y * np.log(p) + (1 - y) * np.log(1 - p))

    grad_w = (p - y) * x
    grad_b = (p - y)

    w = w - (learning_rate * grad_w)
    b = b - (learning_rate * grad_b)

    print(f"Epoch {epoch} | Peluang AI Nebak Lulus : {p*100:.2f}% | Siksaan Error : {error:.4f}")
