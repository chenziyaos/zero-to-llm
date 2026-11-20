# ZTL-08｜模型｜把注意力写成矩阵 · 对比标准线性累加淹没旧信息 vs GDN/KDA 秩-1 Delta 规则精准擦除覆写键值对
import numpy as np

rng = np.random.default_rng(2026)
d = 32
unit = lambda v: v / np.linalg.norm(v)
k_target, v_old = unit(rng.standard_normal(d)), unit(rng.standard_normal(d))
v_raw = rng.standard_normal(d)
v_new = unit(v_raw - (v_raw @ v_old) * v_old)  # 构造与 v_old 正交的新值 v_new

S_add, S_delta = np.zeros((d, d)), np.zeros((d, d))
stream = [(k_target, v_old, 1.0)] + [
    (unit(rng.standard_normal(d)), unit(rng.standard_normal(d)), 0.5) for _ in range(40)
] + [(k_target, v_new, 1.0)]                     # 第 42 步将同一键 k_target 原地更新为 v_new

alpha = np.full(d, 0.99)
for k_t, v_t, beta_t in stream:
    S_add = S_add + np.outer(k_t, v_t)           # 标准线性累加: 只加不减，旧值与背景混杂
    s_decay = alpha[:, None] * S_delta           # GDN/KDA: 逐通道衰减 + 秩-1 Householder 擦旧写新
    S_delta = s_decay - beta_t * np.outer(k_t, k_t @ s_decay) + beta_t * np.outer(k_t, v_t)

out_add, out_delta = S_add.T @ k_target, S_delta.T @ k_target
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
print(f"场景: {len(stream)} 步流（首步旧值+40步扰动+末步新值），第 42 步原地覆写 k_target 为正交 v_new")
print(f"标准累加: 命中新值余弦={cos(out_add, v_new):.3f} | 旧值污染余弦={abs(cos(out_add, v_old)):.3f}")
print(f"GDN/KDA : 命中新值余弦={cos(out_delta, v_new):.4f} | 旧值污染余弦={abs(cos(out_delta, v_old)):.4f}")
