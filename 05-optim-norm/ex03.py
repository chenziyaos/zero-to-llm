# ZTL-05｜数学｜优化、归一化与数值稳定 · 精确复现纯 BF16 下 1e-4 微步更新 1000 步原地踏步，并验证 FP32 主权重与 1x128 分块 FP8 修复效果
import numpy as np

def to_bf16(x_f32):
    arr = np.asarray(x_f32, dtype=np.float32)
    u32 = arr.view(np.uint32)
    lsb = (u32 >> 16) & np.uint32(1)
    return ((u32 + np.uint32(0x7FFF) + lsb) & np.uint32(0xFFFF0000)).view(np.float32)

# 1. 复现 BF16 舍入吞没
w_pure_bf16, w_master_f32 = np.float32(1.0), np.float32(1.0)
for _ in range(1000):
    w_pure_bf16 = to_bf16(w_pure_bf16 - np.float32(1e-4))
    w_master_f32 -= np.float32(1e-4)
print(f"1000 步后: 纯 BF16={float(w_pure_bf16):.6f} (完全停滞) | FP32主副本={float(w_master_f32):.6f} (下发BF16={float(to_bf16(w_master_f32)):.6f})")

# 2. 对比整张量缩放 vs 1x128 分片 FP8 E4M3 (上限 448) 量化误差
rng = np.random.RandomState(2024)
X_base = rng.randn(16, 256).astype(np.float32) * 5.0
X_base[0, 0] = 120.0
X = X_base.copy()
abs_max_global = np.max(np.abs(X))
scale_global = 448.0 / abs_max_global
X_q_global = np.clip(np.round(X * scale_global), -448, 448)
X_dq_global = X_q_global / scale_global
rmse_global = np.sqrt(np.mean((X - X_dq_global)**2))
X_dq_blockwise = np.zeros_like(X)
for col_start in range(0, X.shape[1], 128):
    col_end = min(col_start + 128, X.shape[1])
    block = X[:, col_start:col_end]
    scale_block = 448.0 / (np.max(np.abs(block)) + 1e-8)
    block_q = np.clip(np.round(block * scale_block), -448, 448)
    X_dq_blockwise[:, col_start:col_end] = block_q / scale_block
rmse_blockwise = np.sqrt(np.mean((X - X_dq_blockwise)**2))
print(f"FP8 全局量化 RMSE: {rmse_global:.5f}")
improvement_pct = (1 - rmse_blockwise / rmse_global) * 100
print(f"FP8 1x128 细粒度量化 RMSE: {rmse_blockwise:.5f} (误差降低 {improvement_pct:.0f}%)")
