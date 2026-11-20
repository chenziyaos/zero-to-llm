# ZTL-01｜数学｜符号、向量与矩阵乘法 · 内维规则、手算例子与广播
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])               # (2, 3)
B = np.array([[1, 0],
              [0, 1],
              [1, 1]])                  # (3, 2)
C = A @ B                               # inner dims: 3 == 3
print("A", A.shape, "@ B", B.shape, "->", C.shape)
print(C)

x = np.zeros((4, 10, 128), dtype=np.float32)   # (B, T, in)
W = np.zeros((128, 256), dtype=np.float32)     # (in, out)
print("broadcast:", x.shape, "@", W.shape, "->", (x @ W).shape)
try:
    A @ A                               # (2, 3) @ (2, 3): inner dims 3 != 2
except ValueError:
    print("error: inner dims", A.shape[1], "!=", A.shape[0])
