import numpy as np

def sigmoid(z) :
    return 1 / (1 + np.exp(-z))

def CrossEntropy(y, p) :
    return - (y * np.log(p) + (1 - y) * np.log(1 - p))

# tebakan ai pinter
target_1 = 1
tebakan_1 = 0.9
error_1 = CrossEntropy(target_1, tebakan_1)
print(f"Error kasus 1, (target 1, nebak 0.9) : {error_1}")

# tebakan ai bodoh
target_2 = 0
tebakan_2 = 0.8
error_2 = CrossEntropy(target_2, tebakan_2)
print(f"Error kasus 2, (target 2, nebak 0.8) : {error_2}")





