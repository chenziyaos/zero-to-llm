# ZTL-05｜数学｜优化、归一化与数值稳定 · 验证三统计量在线 Softmax 在极端大分数（1000+）下单遍流式递推与标准路径零误差等价
import math
import numpy as np

# 极端大分数：直接调用 np.exp(1002.0) 在 FP32 与 FP64 下均立即上溢报 Inf
logits = np.array([1000.0, 1001.0, 1002.0, 999.0], dtype=np.float64)
V = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 1.0], [2.0, 4.0]], dtype=np.float64)

# 参照基准：传统三遍扫描安全路径（需预知整行全局最大值 m_global）
m_global = np.max(logits)
probs_ref = np.exp(logits - m_global) / np.sum(np.exp(logits - m_global))
out_ref = probs_ref @ V

# 单遍流式在线路径：片上仅维护标量 m、分母 l 与未归一化输出向量 O
m, l, O = -float("inf"), 0.0, np.zeros(V.shape[1], dtype=np.float64)
for x_i, v_i in zip(logits, V):
    m_new = max(m, float(x_i))
    rescale = math.exp(m - m_new) if m != -float("inf") else 0.0
    weight = math.exp(float(x_i) - m_new)
    l = l * rescale + weight
    O = O * rescale + weight * v_i
    m = m_new

out_online = O / l
print("3-pass Ref :", np.round(out_ref, 6))                 # [2.083004 1.245914]
print("1-pass Onl :", np.round(out_online, 6))              # [2.083004 1.245914]
print("Max abs err:", np.max(np.abs(out_ref - out_online)))  # 0.0
