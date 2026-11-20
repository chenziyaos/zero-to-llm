# ZTL-01｜数学｜符号、向量与矩阵乘法 · 点积方差随头宽增长，以及缩放对 softmax 的影响
import numpy as np

q, k = np.array([1.0, 2, 2]), np.array([3.0, 1, 0])
cos = q @ k / (np.linalg.norm(q) * np.linalg.norm(k))
print(f"q.k = {q @ k:.0f}, |q| = {np.linalg.norm(q):.2f}, |k| = {np.linalg.norm(k):.2f}, "
      f"cos = {cos:.2f}, angle = {np.degrees(np.arccos(cos)):.0f} deg")

rng = np.random.default_rng(0)
n = 100_000
for d in (64, 128, 256):
    a = rng.standard_normal((n, d), dtype=np.float32)
    b = rng.standard_normal((n, d), dtype=np.float32)
    s = (a * b).sum(axis=1)                       # n independent dot products
    print(f"d={d:3d}  std(q.k) = {s.std():5.2f}  sqrt(d) = {np.sqrt(d):5.2f}  "
          f"std(q.k / sqrt(d)) = {(s / np.sqrt(d)).std():.2f}")

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

d, keys, trials = 128, 1024, 2000
Q = rng.standard_normal((trials, 1, d))
K = rng.standard_normal((trials, keys, d))
scores = (Q @ K.transpose(0, 2, 1))[:, 0, :]      # (trials, keys)
for name, z in (("raw", scores), ("scaled", scores / np.sqrt(d))):
    p = softmax(z)
    print(f"{name:6s} mean top-1 weight over {keys} keys = {p.max(axis=-1).mean():.3f}")
