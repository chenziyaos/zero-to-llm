# ZTL-20｜系统｜减少注意力的内存搬运 · FlashMLA 矩阵吸收恒等核验与算术强度对账：零误差免除 71 倍显存展开，解码强度从 6.0 跃至 241.8 FLOPs/B
import math, numpy as np

np.random.seed(2026)
Hq_mla, dc, d_nope, d_rope, dv, T = 128, 512, 128, 64, 128, 256
q_C, q_R = np.random.randn(Hq_mla, d_nope), np.random.randn(Hq_mla, d_rope)
c_KV, k_R = np.random.randn(T, dc), np.random.randn(T, d_rope)
W_UK = np.random.randn(Hq_mla, d_nope, dc) * 0.05
W_UV = np.random.randn(Hq_mla, dv, dc) * 0.05
sc_mla = 1.0 / math.sqrt(d_nope + d_rope)

# 1. 反例显式解压路径：展开 128 头完整键值（每词膨胀至 81,920 B，达潜缓存 1,152 B 的 71.1 倍！）
K_C, V_C = np.einsum("hdk,tk->htd", W_UK, c_KV), np.einsum("hvk,tk->htv", W_UV, c_KV)
lg_exp = (np.einsum("hd,htd->ht", q_C, K_C) + np.einsum("hr,tr->ht", q_R, k_R)) * sc_mla
p_e = np.exp(lg_exp - np.max(lg_exp, axis=-1, keepdims=True))
o_exp = np.einsum("ht,htv->hv", p_e / np.sum(p_e, axis=-1, keepdims=True), V_C)

# 2. FlashMLA 吸收路径：W_UK 左吸成 (128, 576) 稠密查询阵，128 头共享直读 576 维潜缓存跑矩阵乘
q_abs = np.concatenate([np.einsum("hdk,hd->hk", W_UK, q_C), q_R], axis=-1)
kv_lat = np.concatenate([c_KV, k_R], axis=-1)
lg_abs = np.einsum("hd,td->ht", q_abs, kv_lat) * sc_mla
p_a = np.exp(lg_abs - np.max(lg_abs, axis=-1, keepdims=True))
o_tilde = np.einsum("ht,tk->hk", p_a / np.sum(p_a, axis=-1, keepdims=True), c_KV)
o_abs = np.einsum("hvk,hk->hv", W_UV, o_tilde)

ai_qwen38_gqa = (2 * 6 * 256 + 2 * 6 * 256) / (2 * 256 * 2)          # Qwen3.8 6:1 GQA BF16
ai_flashmla   = (2 * 128 * 576 + 2 * 128 * 512) / (576 * 2)          # DeepSeek FlashMLA BF16
print(f"矩阵吸收零误差恒等验证与强度对账")
print(f"误差幅度：{np.max(np.abs(o_exp - o_abs)):.2e}")
print(f"算术强度：Qwen GQA={ai_qwen38_gqa:.1f} vs FlashMLA={ai_flashmla:.1f} FLOPs/B")
