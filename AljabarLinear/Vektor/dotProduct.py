import numpy as np

vektor_a = np.array([2,3])
vektor_b = np.array([4,-1])

fitur_rumah = np.array([3, 120])
bobot_ai = np.array([50, 2])

fitur_hp = np.array([8, 256])
bobot_ai_hp = np.array([10, 3])

hasil_dot = np.dot(vektor_a, vektor_b)
hasil_hp = np.dot(fitur_hp, bobot_ai_hp)
hasil_prediksi = np.dot(fitur_rumah, bobot_ai)

print(f"Hasil docproduct : {hasil_dot}")
print(f"Hasil prediksi : {hasil_prediksi}")
print(f"Hasil prediksi harga hp adalah : {hasil_hp}")