# ZTL-05｜数学｜优化、归一化与数值稳定 · 64 层残差流的尺度增长模拟
import numpy as np

rng = np.random.default_rng(42)
d, L = 256, 64
Ws = [rng.standard_normal((d, d)) / np.sqrt(d) for _ in range(L)]
x0 = rng.standard_normal(d)


def rms(v):
    return np.sqrt(np.mean(v ** 2))


def run(pre_norm):
    x, out = x0.copy(), {}
    for l, W in enumerate(Ws, 1):
        h = x / rms(x) if pre_norm else x
        x = x + W @ h
        if l in (8, 16, 32, 64):
            out[l] = rms(x)
    return out


plain, normed = run(False), run(True)
print('层数   无归一化 RMS     Pre-RMSNorm RMS')
for l in plain:
    print(f'{l:>4} {plain[l]:>14.1f} {normed[l]:>14.2f}')
print(f'第 64 层之比：{plain[64] / normed[64]:.3g} 倍')
