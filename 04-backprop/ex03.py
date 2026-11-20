# ZTL-04｜数学｜链式法则与 softmax-交叉熵的梯度 · 模拟 64/93 层纯串行深网、标准 Pre-Norm 等权残差与 超深架构 Block AttnRes 的梯度保留与抗稀释表现
import math, numpy as np

rng = np.random.default_rng(42)
L, d = 93, 64
Ws = [rng.standard_normal((d, d)) * (0.85 / math.sqrt(d)) for _ in range(L)]
x0 = rng.standard_normal((1, d))
g_top = rng.standard_normal((1, d))
g_top /= np.linalg.norm(g_top)                               # 栈顶第 93 层入口梯度范数固定为 1.0000

def rms_norm(x, eps=1e-6):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)

# 1. 纯串行 93 层网络: 雅可比连乘导致梯度指数级消失
xs = [x0]
for l in range(L):
    xs.append(np.tanh(xs[-1] @ Ws[l].T))
gp, plain_grad = g_top.copy(), {}
for l in range(L - 1, -1, -1):
    gp = (gp * (1.0 - xs[l + 1] * xs[l + 1])) @ Ws[l]
    if L - l in (64, 93):
        plain_grad[L - l] = float(np.linalg.norm(gp))

# 2. 标准 Pre-Norm 等权残差 (权重恒为 1 累加 93 层): 单位阵保住梯度，但隐状态膨胀且冲淡第 0 层
hs = [x0]
for l in range(L):
    hs.append(hs[-1] + np.tanh(rms_norm(hs[-1]) @ Ws[l].T))
swell_std = float(np.linalg.norm(hs[-1]) / np.linalg.norm(x0))
share_std = float(np.sum(x0 * x0) / np.sum(hs[-1] * hs[-1]))

# 3. 超深架构 Block AttnRes (93 层切 8 块): 伪查询零初始化退化均匀平均；下面手设一个对齐第 0 层的伪查询，示意训练后的样子
v_blks = [x0] + [rng.standard_normal((1, d)) for _ in range(8)]
keys = np.vstack([rms_norm(v) for v in v_blks])             # (9, d) 归一化块键表示
alpha_init = np.ones(len(v_blks)) / len(v_blks)             # w_l = 0 起步 -> 严格等权 1/9 = 0.1111
w_learned = 0.40 * keys[0] / math.sqrt(d)                   # 手设（非训练所得）：伪查询与第 0 层特征方向对齐
scores = keys @ w_learned
alpha_learn = np.exp(scores - scores.max()) / np.exp(scores - scores.max()).sum()
h_attn = alpha_learn @ np.vstack(v_blks)
swell_attn = float(np.linalg.norm(h_attn) / np.linalg.norm(x0))

print(f"Configuration: L={L} layers, d={d} hidden dim, 8 blocks")
print(f"Plain serial grad norm | 穿64层={plain_grad[64]:.2e}, 穿93层={plain_grad[93]:.2e}")
print(f"Std Pre-Norm (93L)     | 顶层状态膨胀={swell_std:.2f}x, 第0层能量占比={share_std*100:.2f}%")
print(f"Block AttnRes (93L)    | 顶层状态比值={swell_attn:.2f}x, alpha_(0->top): 起步={alpha_init[0]:.4f} -> 手设对齐后={alpha_learn[0]:.4f}")
