# ZTL-17｜后训练｜先验证奖励再训练 · 32×H100 (8×TP4) 训练案例模型 对比 Qwen3-32B 带宽下限及 V0/V1/V2 事故修复四笔账核算器
import math

N_GPU, BW_GBS, TFLOPS = 32, 3350.0, 989.0          # H100 SXM5: 单卡 3.35 TB/s, 989 TFLOPS BF16
B_P, G, L_IN, L_OUT = 64, 8, 512, 3584             # 每步 64 题 × 8 采样 = 512 序列，均长 4096
seqs_tp4, L_mid = (B_P * G) // 8, L_IN + L_OUT / 2 # 8 组 TP=4 并发解码，单实例 64 条序列

kv_q3  = 64 * 2 * 8 * 128 * 2 / 1e9                # Qwen3-32B:  256 KiB/tok
kv_q38 = 16 * 2 * 4 * 256 * 2 / 1e9                # Qwen3.8-27B: 64 KiB/tok (线性缓存直降 75%)
gdn_st = 48 * 48 * 128 * 128 * 4 / 1e9             # 48层 GDN FP32 定长状态 ≈ 0.151 GB/seq
t_roll_q3  = L_OUT * (2 * 32.76 + seqs_tp4 * L_mid * kv_q3) / (4 * BW_GBS)
t_roll_q38 = L_OUT * (2 * 27.00 + seqs_tp4 * (L_mid * kv_q38 + gdn_st)) / (4 * BW_GBS)
t_learn_q38 = (8.0 * 27.0e9 * B_P * G * (L_IN + L_OUT)) / (N_GPU * TFLOPS * 1e12)
print(f"Rollout下限: Qwen3-32B={t_roll_q3:.2f}s -> Qwen3.8-27B={t_roll_q38:.2f}s | Learner下限={t_learn_q38:.2f}s (总下限 {t_roll_q38+t_learn_q38:.2f}s)")

for tag, r_tr, p_ho, eff, dt, steps in [
    ("V0 事故版(松散正则+弱单测)", 0.910, 0.192, 0.35, 49.5, 180),
    ("V1 仅加固T1~T8反例沙箱",    0.595, 0.520, 0.35, 48.2, 180),
    ("V2 加固沙箱+DAPO+Clip0.28", 0.710, 0.664, 1.00, 54.6, 180),
]:
    se = math.sqrt(p_ho * (1.0 - p_ho) / 1000.0)
    gpu_h = (steps * dt / 3600.0) * N_GPU
    gain = 100.0 * (p_ho - 0.390)
    unit_c = f"{gpu_h / gain:.2f} 卡时/pt" if gain > 0 else "负收益(废片)"
    print(f"{tag:24s} | 训练:{r_tr:.3f} | 留出:{100*p_ho:4.1f}%±{100*se:.1f}% | 有效组:{100*eff:5.1f}% | 耗:{gpu_h:4.1f}h | 单价:{unit_c}")
