# ZTL-21｜系统｜用精度换内存要付什么代价 · SmoothQuant 对角迁移降噪 W8A8 与 AWQ 1% 关键通道放大护航 W4A16 实证
import numpy as np

np.random.seed(7)
# 构造 (64 词元, 256 通道) 激活，第 10、99 通道 (约占 1%) 存在 60 倍固定离群尖峰
X = np.random.normal(0.0, 0.5, size=(64, 256))
salient = [10, 99]
X[:, salient] *= 60.0
W = np.random.normal(0.0, 0.15, size=(256, 128))
Y_ref = np.einsum("nc,ck->nk", X, W)

# 1) W8A8 权活同量: 朴素 Per-Token 激活量化 vs SmoothQuant (alpha = 0.5)
def w8a8_gemm(A, B):
    sa = np.max(np.abs(A), axis=1, keepdims=True) / 127.0
    sb = np.max(np.abs(B), axis=0, keepdims=True) / 127.0
    qa, qb = np.clip(np.round(A / sa), -127, 127), np.clip(np.round(B / sb), -127, 127)
    return np.einsum("nc,ck->nk", qa * sa, qb * sb)

s_sq = np.sqrt(np.max(np.abs(X), axis=0) / np.max(np.abs(W), axis=1))
X_sq, W_sq = X / s_sq[None, :], W * s_sq[:, None]
assert np.allclose(np.einsum("nc,ck->nk", X_sq, W_sq), Y_ref, atol=1e-10)
Y_w8_naive, Y_w8_sq = w8a8_gemm(X, W), w8a8_gemm(X_sq, W_sq)

# 2) W4A16 仅压权重 (g=128): 朴素分组 INT4 vs AWQ 放大保护 1% 关键通道 (s=4.0)
def w4_group(B_mat, g=128):
    Bg = B_mat.reshape(B_mat.shape[0] // g, g, B_mat.shape[1])
    sb = np.maximum(np.max(np.abs(Bg), axis=1, keepdims=True) / 7.0, 1e-8)
    return (np.clip(np.round(Bg / sb), -7, 7) * sb).reshape(B_mat.shape)

s_awq = np.ones(256)
s_awq[salient] = 4.0
Y_w4_naive = np.einsum("nc,ck->nk", X, w4_group(W))
Y_w4_awq = np.einsum("nc,ck->nk", X / s_awq[None, :], w4_group(W * s_awq[:, None]))

rel = lambda Y: np.linalg.norm(Y_ref - Y) / np.linalg.norm(Y_ref) * 100
print(f"W8A8  相对误差 | 朴素Per-Token: {rel(Y_w8_naive):.2f}% | SmoothQuant后: {rel(Y_w8_sq):.2f}% (降噪 {rel(Y_w8_naive)/rel(Y_w8_sq):.1f}x)")
print(f"W4A16 相对误差 | 朴素Group-128: {rel(Y_w4_naive):.2f}% | AWQ护航1%通道: {rel(Y_w4_awq):.2f}% (降噪 {rel(Y_w4_naive)/rel(Y_w4_awq):.1f}x)")
print(f"离群通道保护:第{salient[0]},{salient[1]}列 | 放大系数s_awq={s_awq[salient[0]]:.1f} | 分组救回:{(rel(Y_w4_naive)-rel(Y_w4_awq)):.2f}%误差")
