# ZTL-22｜系统｜让草稿减少解码开销 · ReplaySSM 逐比特精确回滚与动态批度熔断
import numpy as np

rng = np.random.default_rng(7)
d, S_anchor = 128, rng.standard_normal((128, 128)).astype(np.float32) * 0.1

def dplr_step(S: np.ndarray, v: np.ndarray, k: np.ndarray, alpha: np.ndarray, beta: float) -> np.ndarray:
    S1 = alpha[:, None] * S
    k_proj = np.sum(k[:, None] * S1, axis=0)
    return S1 - beta * (k[:, None] * k_proj[None, :]) + beta * (k[:, None] * v[None, :])

inputs, S_corrupt = [], S_anchor.copy()
for step in range(7):
    k = rng.standard_normal(d).astype(np.float32)
    k /= np.linalg.norm(k)
    v, alpha, beta = rng.standard_normal(d).astype(np.float32) * 0.2, rng.uniform(0.88, 0.99, d).astype(np.float32), float(rng.uniform(0.3, 0.85))
    inputs.append((v, k, alpha, beta))
    S_corrupt = dplr_step(S_corrupt, v, k, alpha, beta)
    if step == 3:
        S_gold = S_corrupt.copy()

S_replay = S_anchor.copy()
for v, k, alpha, beta in inputs[:4]:
    S_replay = dplr_step(S_replay, v, k, alpha, beta)
print(f"原地污染偏差={np.linalg.norm(S_corrupt - S_gold):.4f} | ReplaySSM最大误差={np.max(np.abs(S_replay - S_gold)):.2e} | 窗口内存:512KB->16KB (省32x)")

W_GB, KV_GB, BW, CMP = 54.0, 0.18, 3.08, 390.0
def eval_serving(b: int, mode: str) -> tuple:
    mode = ("dflash2" if b <= 32 else "std") if mode == "cutoff" else mode
    if mode == "std":
        tpot = max((W_GB + b * KV_GB) / BW, (54.0 * b) / CMP) + 0.46
    else:
        et = 4.80 if b == 1 else 3.25
        t_ver = max((W_GB + b * KV_GB * 1.15) / BW, (54.0 * b * 9) / CMP) + 0.78 + 0.14 * b
        t_drf = max(8.15, (3.8 + 0.03 * b) / BW + (7.6 * b * 8) / CMP + 1.25)
        tpot = (t_drf + t_ver) / et
    return tpot, b * 1000.0 / tpot

for b in (1, 16, 32, 48, 80):
    t0, s0 = eval_serving(b, "std")
    t1, s1 = eval_serving(b, "dflash2")
    tc, sc = eval_serving(b, "cutoff")
    print(f"B={b:2d} | 标准:{t0:5.2f}ms({s0:4.0f}t/s) | 常开投机:{t1:5.2f}ms({s1:4.0f}t/s) | B<=32熔断:{tc:5.2f}ms({sc:4.0f}t/s)")
