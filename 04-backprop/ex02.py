# ZTL-04｜数学｜链式法则与 softmax-交叉熵的梯度 · 手写线性层与 LoRA 反向梯度：验证 B=0 首步对称破缺，并与自动微分校核误差（< 1e-7）
import numpy as np

rng = np.random.default_rng(42)
N, d_in, d_out, r = 8, 16, 12, 4
X = rng.standard_normal((N, d_in)).astype(np.float32)
W = (rng.standard_normal((d_out, d_in)) * 0.1).astype(np.float32)
b = np.zeros(d_out, dtype=np.float32)
A = (rng.standard_normal((r, d_in)) * 0.1).astype(np.float32)
targets = rng.integers(0, d_out, size=N)

def manual_bwd(X, W, b, A, B, targets):
    H = X @ A.T                                           # (N, r) 低秩中间态
    Z = X @ W.T + H @ B.T + b                             # (N, d_out)
    Zs = Z - Z.max(axis=1, keepdims=True)
    P = np.exp(Zs) / np.exp(Zs).sum(axis=1, keepdims=True)
    G = (P - np.eye(d_out, dtype=X.dtype)[targets]) / N   # 栈顶传入 (p - y) / N
    dX = G @ W + (G @ B) @ A                              # 激活梯度 (N, d_in)
    dW, db = G.T @ X, G.sum(axis=0)                       # 权重与偏置梯度
    dB, dA = G.T @ H, (G @ B).T @ X                       # LoRA 双因子梯度
    return [v.astype(np.float32) for v in (dX, dW, db, dA, dB)]

# 1. 验证首步 B=0 初始化：dB 显著非零打破死锁，dA 严格为 0
print(f"Parameters: N={N}, d_in={d_in}, d_out={d_out}, r={r}")
_, _, _, dA1, dB1 = manual_bwd(X, W, b, A, np.zeros((d_out, r), np.float32), targets)
print(f"Step 1 (B=0): ||dB||_2 = {np.linalg.norm(dB1):.6f}, ||dA||_2 = {np.linalg.norm(dA1):.6f}")

# 2. 验证第 2 步 B!=0：与 torch.autograd (或 FP64 中心差分) 逐项比对最大绝对误差
B2 = (rng.standard_normal((d_out, r)) * 0.05).astype(np.float32)
grads_m = manual_bwd(X, W, b, A, B2, targets)
try:
    import torch, torch.nn.functional as F
    xt, wt, bt, at, b2t = [torch.tensor(v, requires_grad=True) for v in (X, W, b, A, B2)]
    F.cross_entropy(xt @ (wt + b2t @ at).T + bt, torch.tensor(targets)).backward()
    refs = [t.grad.numpy() for t in (xt, wt, bt, at, b2t)]
except ImportError:
    p64 = [v.astype(np.float64) for v in (X, W, b, A, B2)]
    def loss64():
        x_, w_, b_, a_, b2_ = p64
        z = x_ @ (w_ + b2_ @ a_).T + b_
        m = z.max(axis=1, keepdims=True)
        return float((-z[np.arange(N), targets] + m[:, 0] + np.log(np.exp(z - m).sum(axis=1))).mean())
    refs, eps = [], 1e-6
    for arr in p64:
        g = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            orig = arr[idx]; arr[idx] = orig + eps; lp = loss64()
            arr[idx] = orig - eps; lm = loss64(); arr[idx] = orig
            g[idx] = (lp - lm) / (2 * eps)
        refs.append(g.astype(np.float32))

max_err = max(np.max(np.abs(m - r_)) for m, r_ in zip(grads_m, refs))
print(f"Step 2 max abs error across (dX, dW, db, dA, dB): {max_err:.2e}")
assert max_err < 1e-7
