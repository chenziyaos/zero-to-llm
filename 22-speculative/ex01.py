# ZTL-22｜系统｜让草稿减少解码开销 · 修正拒绝采样 10 万次蒙特卡洛随机模拟核验
import random

def speculative_sample_one(p: list[float], q: list[float], rng: random.Random):
    # 1. 草稿提议: 按截断归一化后的草稿分布 q(x) 抽取候选词 x
    u_q, cdf, x = rng.random(), 0.0, len(q) - 1
    for i, qi in enumerate(q):
        cdf += qi
        if u_q < cdf:
            x = i
            break
    # 2. 概率比削峰: 以 min(1, p(x)/q(x)) 概率直接放行候选词
    if rng.random() < min(1.0, p[x] / max(q[x], 1e-12)):
        return x, True
    # 3. 残差正部补偿: 拒绝时严格从 max(0, p - q) 归一化分布补采 (绝不重抽 p!)
    pos_diff = [max(0.0, pi - qi) for pi, qi in zip(p, q)]
    z = sum(pos_diff)
    if z <= 1e-12:  # 浮点下溢工程兜底: 两分布完全重合时安全回退
        return x, False
    u_r, cdf = rng.random() * z, 0.0
    for i, di in enumerate(pos_diff):
        cdf += di
        if u_r < cdf:
            return i, False
    return len(p) - 1, False

p, q = [0.50, 0.40, 0.10], [0.30, 0.20, 0.50]
alpha_theory = sum(min(pi, qi) for pi, qi in zip(p, q))
print(f"目标分布 p={p}, 草稿分布 q={q}")
print(f"理论接受率 α={alpha_theory:.4f}")
rng, N, counts, acc = random.Random(42), 100_000, [0, 0, 0], 0
for _ in range(N):
    tok, ok = speculative_sample_one(p, q, rng)
    counts[tok] += 1
    acc += int(ok)
emp_p = [c / N for c in counts]
tvd = 0.5 * sum(abs(ep - pi) for ep, pi in zip(emp_p, p))
print(f"实测接受率={acc/N:.4f} | 实测分布={[round(v,4) for v in emp_p]}, TVD={tvd:.4f}")
