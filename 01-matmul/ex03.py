# ZTL-01｜数学｜符号、向量与矩阵乘法 · Qwen3-32B 一层的参数量与预填充运算量
H, D, hq, hkv, dh, L = 5120, 25600, 64, 8, 128, 64   # Qwen3-32B config
shapes = {
    "q_proj":    (hq * dh, H),
    "k_proj":    (hkv * dh, H),
    "v_proj":    (hkv * dh, H),
    "o_proj":    (H, hq * dh),
    "gate_proj": (D, H),
    "up_proj":   (D, H),
    "down_proj": (H, D),
}
total = 0
for name, (out_f, in_f) in shapes.items():
    n = out_f * in_f
    total += n
    print(f"{name:10s} weight {str((out_f, in_f)):15s} {n / 1e6:7.2f} M")
print(f"one layer {total / 1e6:.1f} M, {L} layers {total * L / 1e9:.2f} B")
print(f"prefill T=1024, linear layers only: {2 * total * L * 1024 / 1e12:.1f} TFLOP")
