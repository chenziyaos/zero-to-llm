# ZTL-18｜后训练｜多想一会儿是否值得 · Codex 连乘无偏 Pass@k 方差核验与双错误分布 Maj@k 蒙特卡洛对账
from collections import Counter
import numpy as np

def pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Chen et al. (Codex) 连乘无偏估计器，规避大阶乘溢出与插件式低估偏差"""
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))

def sim_maj_at_k(probs, k: int, trials: int = 50000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(probs), size=(trials, k), p=probs)
    wins = 0.0
    for row in draws:
        cnt = Counter(row)
        mx = max(cnt.values())
        top = [ans for ans, v in cnt.items() if v == mx]
        if 0 in top:  # 下标 0 为唯一真解，平票时等概率随机打破
            wins += 1.0 / len(top)
    return wins / trials

probs_a = [0.25] + [0.05] * 15           # 分布甲：真解 25%，15 种离散错解各占 5%
probs_b = [0.25, 0.45] + [0.05] * 6      # 分布乙：真解 25%，共性陷阱错解占 45%
for tag, pr in [("分布甲-错解离散", probs_a), ("分布乙-共性陷阱", probs_b)]:
    for k in (1, 5, 16, 64):
        pk = 1.0 - (1.0 - pr[0]) ** k
        mk = sim_maj_at_k(pr, k)
        print(f"{tag} k={k:2d} | 真Pass@k={pk:.4f} | Maj@k={mk:.4f}")

rng = np.random.default_rng(7)
raw_k5 = (rng.random((2000, 5)) < 0.25).any(axis=1).astype(float)
unb_n64 = [pass_at_k_unbiased(64, int(c), 5) for c in rng.binomial(64, 0.25, size=2000)]
print("直抽 n=5  估 Pass@5 均值/标准差:", round(float(raw_k5.mean()), 4), round(float(raw_k5.std()), 4))
print("池化 n=64 估 Pass@5 均值/标准差:", round(float(np.mean(unb_n64)), 4), round(float(np.std(unb_n64)), 4))
