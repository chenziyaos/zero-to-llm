# ZTL-01｜数学｜符号、向量与矩阵乘法 · 换括号的开销与 LoRA 合并的等价性
import numpy as np

def flops(m, k, n):                     # one (m, k) @ (k, n) matmul
    return 2 * m * k * n

H, D, r = 5120, 25600, 16               # Qwen3-32B up_proj, LoRA rank 16
# one decoding token x: (1, H); low-rank update A: (H, r), B: (r, D)
merge_first = flops(H, r, D) + flops(1, H, D)   # x (A B): build A B first
narrow_first = flops(1, H, r) + flops(1, r, D)  # (x A) B: pass the narrow edge first
print(f"x(AB): {merge_first / 1e9:.2f} GFLOP   (xA)B: {narrow_first / 1e6:.2f} MFLOP")
print(f"ratio: {merge_first / narrow_first:.0f}x")

full, lora = H * D, r * (H + D)
print(f"params full {full / 1e6:.1f} M, LoRA {lora / 1e3:.1f} K ({lora / full:.3%})")
print(f"base matmul per token {flops(1, H, D) / 1e6:.1f} MFLOP, "
      f"unmerged LoRA adds {narrow_first / flops(1, H, D):.2%}")

rng = np.random.default_rng(0)
W = rng.standard_normal((64, 48))
A = rng.standard_normal((64, 4)); B = rng.standard_normal((4, 48))
x = rng.standard_normal((3, 64))
merged = x @ (W + A @ B)                # merge once before serving
side = x @ W + (x @ A) @ B              # keep the side branch
print("rank(A @ B) =", np.linalg.matrix_rank(A @ B),
      "| max |merged - side| =", f"{np.abs(merged - side).max():.1e}")
