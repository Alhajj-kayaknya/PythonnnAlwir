import numpy as np

def fungsi_error (x) :
    return 2 * (x ** 4)

def turunan_error (x) :
    return 8 * (x ** 3)

posisi_x = 2

errorSekarang = fungsi_error(posisi_x)
GradientSekarang = turunan_error(posisi_x)

print(f"Posisi X = {posisi_x}")
print(f"Besar Error = {errorSekarang}")
print(f"Gradient = {GradientSekarang}")