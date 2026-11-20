# ZTL-05｜数学｜优化、归一化与数值稳定 · 手写全局 L2 保向裁剪、解耦 AdamW（对齐 PyTorch）与 MuonClip 的 QK-Clip 权重钳制验证
import math
import numpy as np

# 1. 全局 L2 范数裁剪 vs 逐元素截断：验证方向保真度
g_raw = np.array([0.6, 8.0])
c_scale = min(1.0, 1.0 / (np.linalg.norm(g_raw) + 1e-6))
g_norm_clip = g_raw * c_scale
g_val_clip = np.clip(g_raw, -1.0, 1.0)
cos_bad = (g_raw @ g_val_clip) / (np.linalg.norm(g_raw) * np.linalg.norm(g_val_clip))
print("Global clip:", np.round(g_norm_clip, 4), "| Value clip angle err:", round(math.degrees(math.acos(cos_bad)), 2), "deg")

# 2. 手写单步 AdamW（拆分二维权重衰减组与一维 RMSNorm 免衰减组）
def adamw_step(theta, gamma, m, v, g, lr=0.001, beta1=0.9, beta2=0.999, wd=0.01, t=1):
    m_new = beta1 * m + (1 - beta1) * g
    v_new = beta2 * v + (1 - beta2) * g**2
    m_hat = m_new / (1 - beta1**t)
    v_hat = v_new / (1 - beta2**t)
    theta_new = theta - lr * (m_hat / (np.sqrt(v_hat) + 1e-8) + wd * theta)
    g_gamma = np.ones_like(gamma) * 0.001
    m_gamma = beta1 * np.zeros_like(gamma) + (1 - beta1) * g_gamma
    v_gamma = beta2 * np.zeros_like(gamma) + (1 - beta2) * g_gamma**2
    m_hat_gamma = m_gamma / (1 - beta1**t)
    v_hat_gamma = v_gamma / (1 - beta2**t)
    gamma_new = gamma - lr * (m_hat_gamma / (np.sqrt(v_hat_gamma) + 1e-8))
    return theta_new, gamma_new, m_new, v_new

theta1 = np.array([2.0, -0.5])
gamma1 = np.array([1.0, 1.0])
m1 = np.zeros_like(theta1)
v1 = np.zeros_like(theta1)
g1 = np.array([0.1, 0.1])
theta_upd, gamma_upd, _, _ = adamw_step(theta1, gamma1, m1, v1, g1, t=1)
print("AdamW t=1 theta:", np.round(theta_upd, 6), "| gamma:", np.round(gamma_upd, 3))

# 3. Kimi K2 QK-Clip 原理核验：当某头最大内积 S_max=144 越过阈值 100 时原地回压 Q, K 权重
S_max_before = 144.0
threshold = 100.0
qk_clip_factor = np.sqrt(threshold / S_max_before)
S_max_after = S_max_before * (qk_clip_factor ** 2)
print(f"QK-Clip factor={qk_clip_factor:.4f} -> rescaled max logit={S_max_after:.1f}")
