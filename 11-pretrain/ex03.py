# ZTL-11｜训练｜从语料到一轮小预训练 · 一轮 0.5B×10B 预训练四档配置真 MFU、8×H100 墙钟耗时与成本自动复算核验脚本
P_ACT, D_VALID = 0.50e9, 10.0e9   # 0.5B 激活参数，10B 无污染有效训练词元
PEAK_TF, USD_HR = 989.0, 3.0      # H100 SXM5 BF16 稠密峰值 989 TFLOPS，3 美元/卡时

runs = [
    ("1. FP32 + 补PAD",             31000,  0.581),
    ("2. BF16 + 补PAD",            142300,  0.581),
    ("3. BF16 + 双隔离Packing",     142800,  1.000),
    ("4. Packing + 编译融合 + WSD", 168700,  1.000),
]
for tag, raw_slot_s, valid_frac in runs:
    v_tps = raw_slot_s * valid_frac
    eff_tf = 6.0 * P_ACT * v_tps / 1e12
    mfu = eff_tf / PEAK_TF * 100.0
    gpu_h = D_VALID / v_tps / 3600.0
    print(f"{tag:26s} | 有效词/s:{v_tps:7.0f} | 真MFU:{mfu:5.1f}% | {gpu_h:5.1f}卡时 (8卡{gpu_h/8:4.2f}h) | ${gpu_h*USD_HR:5.1f}")
