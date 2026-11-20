# ZTL-04｜数学｜链式法则与 softmax-交叉熵的梯度 · 验算三大模型栈顶词表显存尖峰，并核验分块融合线性交叉熵（Fused Chunked Linear-CE）与全量物化的梯度等价性
import numpy as np

# 1. 三大案例模型在微批词元 N=32,768 下的栈顶输出头缓冲字节精算 (GB = 10^9 B)
N_tok = 32768
for name, V in [("稀疏 671B", 129280), ("稠密 32B", 151936), ("稠密 27B", 248320)]:
    dual_gb = 2 * N_tok * V * 4 / 1e9          # FP32 logits + FP32 (p - y) 梯度双缓冲
    inplace_gb = N_tok * V * (2 + 4) / 1e9     # BF16 logits + 单份 FP32 原地缓冲
    print(f"{name:11s} (V={V}): dual_FP32={dual_gb:.2f} GB | BF16+FP32_inplace={inplace_gb:.2f} GB")
chunk_gb = 512 * 248320 * 4 / 1e9
print(f"Qwen3.8-27B Fused Chunked (C=512): {chunk_gb:.2f} GB (直降 {65.0956 / chunk_gb:.0f} 倍!)")

# 2. 验证分块融合流式覆写 (仅分配 C×V 缓冲) 与全量物化路径的损失及梯度等价性
rng = np.random.default_rng(2026)
N, d, V, C = 128, 64, 512, 32
X = rng.standard_normal((N, d)).astype(np.float32)
W = (rng.standard_normal((V, d)) * 0.05).astype(np.float32)
y = rng.integers(0, V, size=N)

Z = X @ W.T
m = Z.max(axis=1, keepdims=True)
lse = m + np.log(np.exp(Z - m).sum(axis=1, keepdims=True))
loss_ref = float((-Z[np.arange(N), y] + lse[:, 0]).mean())
G = np.exp(Z - lse); G[np.arange(N), y] -= 1.0; G /= N
dX_ref, dW_ref = G @ W, G.T @ X

loss_sum, dX_chk, dW_chk = 0.0, np.empty_like(X), np.zeros_like(W)
for s in range(0, N, C):
    e = min(s + C, N)
    xc, yc = X[s:e], y[s:e]
    buf = xc @ W.T                                          # 任意时刻仅物化 (C, V) 微块缓冲
    mc = buf.max(axis=1, keepdims=True)
    lsec = mc + np.log(np.exp(buf - mc).sum(axis=1, keepdims=True))
    loss_sum += float((-buf[np.arange(e - s), yc] + lsec[:, 0]).sum())
    np.exp(buf - lsec, out=buf)                             # 原地转写 p_c，零额外大张量分配
    buf[np.arange(e - s), yc] -= 1.0; buf /= N              # 原地写为 (p_c - y_c) / N
    dX_chk[s:e] = buf @ W
    dW_chk += buf.T @ xc

print(f"Max diff | loss={abs(loss_sum/N - loss_ref):.2e}, dX={np.max(np.abs(dX_chk - dX_ref)):.2e}, dW={np.max(np.abs(dW_chk - dW_ref)):.2e}")
assert abs(loss_sum / N - loss_ref) < 1e-6 and np.max(np.abs(dW_chk - dW_ref)) < 1e-7
