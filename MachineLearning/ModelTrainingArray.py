import numpy as np

x = np.array([2, 4, 6, 8, 10])
y = np.array([0, 0, 0, 1, 1])

w = 0.0
b = 0.0
learning_rate = 0.1

print("=== MULAI TRAINING AI (DATA 5 ORANG) ===")

for epoch in range(1, 1001) :
    z = (w * x) + b

    p = 1 / (1 + np.exp(-z))
    error = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    grad_w = np.mean((p - y) * x)
    grad_b = np.mean(p - y)

    w = w - (learning_rate * grad_w)
    b = b - (learning_rate * grad_b)

    if epoch % 200 == 0 :
        print(f"Epoch {epoch} | Siksaan Error Rata-Rata : {error:.4f} | w : {w:.4f} | b : {b:.4f}")

print("\n=== Hasil Akhir ===")
print("Peluang Lulus 5 Murid (p) : ", np.round(p, 2))


x_ujian = np.array([7, 3])

z_ujian = (w * x_ujian) + b
p_ujian = 1 / (1 + np.exp(-z_ujian))

print("\n=== SOAL UJIAN MURID BARU ===")
print(f"Jam Belajar : {x_ujian}")
print(f"Peluang Lulus : {np.round(p_ujian, 2)}")



x_ninja = np.array([[10, 12, 13, 14, 0.5], [1, 2, 3, 4, 5]])

z_ninja = (w * x_ninja) + b
p_ninja = 1 / (1 + np.exp(-z_ninja))

print("\n=== Hasil Ujian Ninja ===")
print(f"Jam Belajar : {x_ninja}")
print(f"Peluang Lulus : {np.round(p_ninja, 5)}")














