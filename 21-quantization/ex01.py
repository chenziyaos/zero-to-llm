# ZTL-21｜系统｜用精度换内存要付什么代价 · Per-Tensor 单刻度坍缩 vs Group-128 分组量化误差及 Qwen3.8-27B 真实显存拆账
import numpy as np

GIB = 1024 ** 3
# 1) Qwen3.8-27B 分模块真实显存核算 (vocab=248,320 非绑定头尾保8b, GDN门控与Norm保16b)
p_total = 27.00e9
p_vocab = 2 * 248320 * 5120                  # 2.5428B 头尾词表参数
p_gate_norm = 0.12e9                         # 0.1200B GDN门控/卷积核与归一化
p_linear = p_total - p_vocab - p_gate_norm   # 24.3372B 主干线性层
b_awq = 4.0 + (16.0 + 4.0) / 128.0           # 4.15625 bits

naive_gb = p_total * 0.5 / 1e9
real_gb = (p_linear * (b_awq / 8.0) + p_vocab * 1.0 + p_gate_norm * 2.0) / 1e9
real_gib = real_gb * 1e9 / GIB
gdn_gib = (48 * 48 * 128 * 128 * 4) / GIB    # 48层GDN FP32定长状态 = 0.14 GiB
kv_free_gib = 24.0 - real_gib - 2.20 - gdn_gib
max_ctx_k = int((kv_free_gib * GIB) // (16 * 2 * 4 * 256 * 2)) // 1024

# 2) 单个极值 3.60 撑废 Per-Tensor INT4 vs Group-128 隔离保护实验
np.random.seed(42)
w = np.random.normal(0.0, 0.08, size=512).astype(np.float32)
w[17] = 3.60
s_t = np.max(np.abs(w)) / 7.0
q_t = np.clip(np.round(w / s_t), -7, 7)
mse_t = np.mean(np.square(w - q_t * s_t))

w_g = w.reshape(-1, 128)
w_min, w_max = w_g.min(axis=1, keepdims=True), w_g.max(axis=1, keepdims=True)
s_g = np.maximum((w_max - w_min) / 15.0, 1e-8).astype(np.float16).astype(np.float32)
z_g = np.clip(np.round(-w_min / s_g), 0, 15)
w_hat = ((np.clip(np.round(w_g / s_g) + z_g, 0, 15) - z_g) * s_g).reshape(-1)
mse_g = np.mean(np.square(w - w_hat))

print(f"Qwen3.8-27B | 拍脑袋:{naive_gb:.2f}GB | 真实拆账:{real_gb:.2f}GB ({real_gib:.2f}GiB) | 4090净KV池:{kv_free_gib:.2f}GiB ({max_ctx_k}K词)")
print(f"误差对比    | Per-Tensor归零率={np.mean(q_t==0)*100:.1f}% MSE={mse_t:.6f} | Group-128有效位宽={b_awq:.5f}b MSE={mse_g:.6f}")
print(f"单刻度极值撑废:{np.sum(q_t==0)}/{len(q_t)}权重 | 分组隔离救回:{(np.mean(q_t==0)-np.mean(w_hat==0))*100:.1f}%权重")
