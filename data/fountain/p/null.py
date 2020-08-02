import numpy as np
from scipy.linalg import null_space

P = np.array([[2246.17,-2208.64,-62.134,24477.4],[-316.161,-1091.08,2713.64,-8334.18],[-0.269944,-0.961723,-0.0471142,-7.01217]])

print(P)

C = null_space(P)
C = C/C[3]
print(C)
