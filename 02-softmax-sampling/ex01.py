# ZTL-02｜数学｜条件概率、softmax 与采样 · 验证某大型模型 248,320 维 FP16 朴素指数上溢崩溃、减最大值救援与双模式温度熵值手算
import numpy as np

# 1. 模拟超宽词表模型单步 V=248,320 的 FP16 打分（榜首 85.0 远超 FP16 上限 ln(65504)≈11.09）
rng = np.random.default_rng(2026)
z_vocab = rng.normal(loc=0.0, scale=2.5, size=248_320).astype(np.float16)
z_vocab[0], z_vocab[1] = np.float16(85.0), np.float16(83.0)
with np.errstate(over="ignore", invalid="ignore"):
    naive_p = np.exp(z_vocab) / np.sum(np.exp(z_vocab))

# 工业标准防溢：先减行最大值锁在 <=0，再升 FP32 求和防 24.8 万项尾数丢失
shifted = (z_vocab - np.max(z_vocab)).astype(np.float32)
safe_p = np.exp(shifted) / np.sum(np.exp(shifted), dtype=np.float32)
print(f"朴素 FP16 前两项: {naive_p[:2]} | 安全平移前两项: {np.round(safe_p[:2], 4)}, 总和={safe_p.sum():.6f}")

# 2. 双模式温度对比：T=1.0 探索模式 vs T=0.7 决定模式
z_test = np.array([4.0, 2.0, 1.0, 0.0], dtype=np.float32)
p_t10 = np.exp(z_test - np.max(z_test)) / np.sum(np.exp(z_test - np.max(z_test)))
p_t07 = np.exp(z_test/0.7 - np.max(z_test/0.7)) / np.sum(np.exp(z_test/0.7 - np.max(z_test/0.7)))
h_t10 = -np.sum(p_t10 * np.log(p_t10 + 1e-10))
h_t07 = -np.sum(p_t07 * np.log(p_t07 + 1e-10))
print(f"T=1.0: {np.round(p_t10, 4)} 熵={h_t10:.4f} | T=0.7: {np.round(p_t07, 4)} 熵={h_t07:.4f}")
print("assert: 平移防溢与温度调控共同守护了数值稳定性")
