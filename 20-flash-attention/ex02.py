# ZTL-20｜系统｜减少注意力的内存搬运 · 案例模型 全局层（24Q/4KV×256）Split-K=16 局部三元组结合律归约零误差验证（16 → 256 线程块）
import math, numpy as np

np.random.seed(2026)
B, Hq, Hkv, Lkv, dh, Sk = 4, 24, 4, 4096, 256, 16                  # 对齐 案例模型 6:1 GQA 形状
q_dec = np.random.randn(B, Hkv, Hq // Hkv, dh).astype(np.float32)  # 同组 6 个 Q 头打包进单块共享 KV
k_cache = np.random.randn(B, Hkv, Lkv, dh).astype(np.float32)
v_cache = np.random.randn(B, Hkv, Lkv, dh).astype(np.float32)
sc = 1.0 / math.sqrt(dh)

# 1. 无 Split-K 基准：全卡仅 B*Hkv = 16 个线程块串行扫完整条历史（H100 上 116 个 SM 空转）
logits = np.einsum("bhgd,bhkd->bhgk", q_dec, k_cache) * sc
p_ref = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
o_ref = np.einsum("bhgk,bhkd->bhgd", p_ref / np.sum(p_ref, axis=-1, keepdims=True), v_cache)

# 2. Split-K=16 路径：沿历史轴切成 256 个线程块独立产出局部三元组 (m_s, l_s, O_tilde_s)
chunk = Lkv // Sk
m_list, l_list, ot_list = [], [], []
for s in range(Sk):
    ks, vs = k_cache[:, :, s * chunk:(s + 1) * chunk], v_cache[:, :, s * chunk:(s + 1) * chunk]
    sc_s = np.einsum("bhgd,bhkd->bhgk", q_dec, ks) * sc
    m_s = np.max(sc_s, axis=-1, keepdims=True)
    p_s = np.exp(sc_s - m_s)
    m_list.append(m_s)
    l_list.append(np.sum(p_s, axis=-1, keepdims=True))
    ot_list.append(np.einsum("bhgk,bhkd->bhgd", p_s, vs))

# 3. 轻量对数级合并归约：利用在线三元组结合律一步还原精确全局输出
M_st, L_st, OT_st = np.stack(m_list, axis=0), np.stack(l_list, axis=0), np.stack(ot_list, axis=0)
m_glob = np.max(M_st, axis=0, keepdims=True)
rescale = np.exp(M_st - m_glob)
o_splitk = np.sum(rescale * OT_st, axis=0) / np.sum(rescale * L_st, axis=0)
print(f"Split-K 沿历史轴切分唤醒 SM")
print(f"活跃线程块：{B*Hkv} -> {B*Hkv*Sk}")
print(f"三元组合并最大误差：{np.max(np.abs(o_ref - o_splitk)):.2e}")
