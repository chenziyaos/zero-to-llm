# ZTL-08｜模型｜把注意力写成矩阵 · 25 行零依赖 NumPy 核验：证明 DeepSeek MLA 显式键值展开态与解码矩阵吸收态达到 10^-16 机器精度等价
import numpy as np

rng = np.random.default_rng(7)
S, H, d_m, d_c, d_k, d_r = 8, 4, 64, 16, 12, 8
c_kv = rng.standard_normal((S, d_m)) @ (rng.standard_normal((d_m, d_c)) * 0.1)
k_r = rng.standard_normal((S, d_r))                  # HBM 仅存潜向量 c_kv(S,d_c) 与共享旋转键 k_r(S,d_r)
W_UK = rng.standard_normal((H, d_c, d_k)) * 0.1
W_UV = rng.standard_normal((H, d_c, d_k)) * 0.1
W_O = rng.standard_normal((H, d_k, d_m)) * 0.1
q_c, q_r = rng.standard_normal((H, d_k)), rng.standard_normal((H, d_r))
scale = 1.0 / np.sqrt(d_k + d_r)

# 路径 A (Prefill 算力友好展开态): 还原全部历史词的 K_C, V_C 形状 (H, S, d_k)
K_C = np.einsum("sd,hdk->hsk", c_kv, W_UK)
V_C = np.einsum("sd,hdk->hsk", c_kv, W_UV)
logits_a = (np.einsum("hk,hsk->hs", q_c, K_C) + q_r @ k_r.T) * scale
p_a = np.exp(logits_a - logits_a.max(axis=-1, keepdims=True))
p_a /= p_a.sum(axis=-1, keepdims=True)
out_a = np.einsum("hs,hsk,hkm->m", p_a, V_C, W_O)

# 路径 B (Decode 访存友好吸收态): W_UK 左吸入查询侧，W_UV 右吸入 W_O，全程免展开历史 KV！
q_abs = np.einsum("hk,hdk->hd", q_c, W_UK)           # 吸收后查询形状 (H, d_c)
logits_b = (q_abs @ c_kv.T + q_r @ k_r.T) * scale
p_b = np.exp(logits_b - logits_b.max(axis=-1, keepdims=True))
p_b /= p_b.sum(axis=-1, keepdims=True)
out_b = np.einsum("hd,hdk,hkm->m", p_b @ c_kv, W_UV, W_O)

err = np.max(np.abs(out_a - out_b))
print(f"模型尺寸: S={S} 序列词数, H={H} 头数, d_c={d_c} 潜缓存维, d_r={d_r} 旋转维")
print(f"展开态每词元素:{2*H*d_k + H*d_r} | MLA缓存元素:{d_c + d_r} | 省存比例:{100*(1-( d_c + d_r)/(2*H*d_k + H*d_r)):.1f}%")
print(f"最大绝对误差:{err:.2e} (双精度舍入下限~1e-16)")
assert err < 1e-14  # 实跑输出: 展开态每词元素:128 | MLA缓存元素:24 | 最大绝对误差:3.89e-16
