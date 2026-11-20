# ZTL-01｜数学｜符号、向量与矩阵乘法 · 教科书写法与批量写法等价
import numpy as np

rng = np.random.default_rng(0)
W = rng.standard_normal((256, 128))     # stored like nn.Linear: (out, in)
x = rng.standard_normal((4, 128))       # 4 row vectors, (B, in)

y_rows = x @ W.T                        # batch form: y = x W^T
y_cols = (W @ x.T).T                    # textbook form: y = W x, one column per sample
print("W", W.shape, "W.T", W.T.shape, "y", y_rows.shape)
print("max |x W^T - (W x)^T| =", np.abs(y_rows - y_cols).max())
print("W.T is a view, no copy:", np.shares_memory(W, W.T))
