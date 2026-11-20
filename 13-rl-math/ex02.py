# ZTL-13｜后训练｜奖励、策略梯度与 DPO 的数学 · 玻尔兹曼闭式最优分布、配分函数 Z(x) 精确对消与 DPO 错判样本动态梯度权重数值验证
import math
import numpy as np

pi_ref = np.array([0.50, 0.30, 0.15, 0.05])
r_true = np.array([1.20, 0.40, -0.20, 2.00])
beta = 0.50
# 1. 解析计算玻尔兹曼闭式最优策略 pi* 与配分函数 Z(x)
unnorm = pi_ref * np.exp(r_true / beta)
Z_x = unnorm.sum()
pi_star = unnorm / Z_x
r_implicit = beta * np.log(pi_star / pi_ref)                 # 省去不可算常数 +beta * ln Z(x)
print(f"配分函数 Z(x) = {Z_x:.4f} | 最优分布 pi* = {pi_star.round(4).tolist()}")
print(f"真实差 r0-r1  = {r_true[0]-r_true[1]:.4f} | 隐式差对消 Z(x) = {r_implicit[0]-r_implicit[1]:.4f}")

# 2. 核验当前策略 pi_theta 在已排对样本 (y0≻y2) 与正判错样本 (y3≻y1) 上的自适应梯度权重
pi_theta = np.array([0.65, 0.30, 0.03, 0.02])
log_pi, log_ref = np.log(pi_theta), np.log(pi_ref)
for label, w, l in [("已排对样本 (y0≻y2)", 0, 2), ("正判错样本 (y3≻y1)", 3, 1)]:
    margin = beta * ((log_pi[w] - log_ref[w]) - (log_pi[l] - log_ref[l]))
    prob_ok = 1.0 / (1.0 + math.exp(-margin))
    grad_weight = 1.0 - prob_ok                              # 恰好等于 sigma(r_hat(y_l) - r_hat(y_w))
    print(f"{label} -> 边际:{margin:+.4f} | 错判梯度权重:{grad_weight:.4f} | 损失:{-math.log(prob_ok):.4f}")
