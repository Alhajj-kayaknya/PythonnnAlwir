import numpy as np

x = 10.0
y = 0.0
w = 0.5
b = 0.0

print("=== Bongkar Mesin Chain Rules ===")

# 1. Forward Pass (Jalan Maju)
z = (w * x) + b
p = 1 / (1 + np.exp(-z))

print(f"Peluang Tebakan Awal : {p:.4f}")

# 2. Backward Pass (Chain Rule / Jalan Mundur)
# Mesin A: Turunan Error (Cross-Entropy) terhadap output sigmoid (p)
turunan_error_ke_p = (p - y) / (p * (1 - p))

# Mesin B: Turunan Sigmoid (p) terhadap z
turunan_p_ke_z = p * (1 - p)

# Mesin C: Turunan z (z = wx + b) terhadap w
turunan_z_ke_w = x

grad_w_chain = turunan_error_ke_p * turunan_p_ke_z * turunan_z_ke_w

grad_w_cheat = (p - y) * x
print(f"Gradint w pake chain rule utuh : {grad_w_chain:.4f}")
print(f"Gradint w cheat chain rule : {grad_w_cheat:.4f}")