# ZTL-13｜后训练｜奖励、策略梯度与 DPO 的数学 · 单机 8×H100 训练稠密混合 27B：四模型 PPO、离线 DPO 与共置 SGLang 睡眠唤醒 GRPO 对账脚本
PSI_B, N_GPU, HBM_MAX = 27.0, 8, 85.90       # 27B 参数量 与 H100 单卡容量(GB)
PEAK_TFLOPS, HBM_BW_GBS = 989.0, 3350.0

# 1. 标准四模型 PPO 静态底盘 vs 预缓存 ref 的离线 DPO (每步 16 对 = 32 条 × 2048 词元)
m_ppo_static = 36.0 * PSI_B / N_GPU
m_dpo_peak = 16.0 * PSI_B / N_GPU + 10.20    # 54.0 GB 静态分片 + 10.2 GB 重算激活与通信缓冲
t_dpo = (6.0 * PSI_B * 1e9 * (32 * 2048)) / (N_GPU * PEAK_TFLOPS * 1e12 * 0.40)

# 2. 共置 SGLang 睡眠唤醒 GRPO (8 题 × G=8 = 64 条长链, 最长 3072, 均长 2400, 在途均长 1536)
w_tp8_gb = (2.0 * PSI_B) / N_GPU             # 采样态休眠 12Ψ 优化器, TP=8 权重仅 6.75 GB/卡
kv_gqa_gb = (64 * 1536 * 65536) / N_GPU / 1e9          # 16 层 Gated GQA @ 64 KiB/tok
gdn_state_gb = (64 * 144 * 1024 * 1024) / N_GPU / 1e9  # 48 层 GDN FP32 定长状态 @ 144 MiB/序列
t_roll = 3072 * ((w_tp8_gb + kv_gqa_gb + gdn_state_gb) / (HBM_BW_GBS * 0.72)) + 1.15
tok_grpo = 64 * 2400
t_upd = (2.0 * PSI_B * 1e9 * tok_grpo) / (N_GPU * PEAK_TFLOPS * 1e12 * 0.54) + (
        (6.0 * PSI_B * 1e9 * tok_grpo) / (N_GPU * PEAK_TFLOPS * 1e12 * 0.40))
m_grpo_peak = 16.0 * PSI_B / N_GPU + 12.40   # 训练态释放 SGLang 池, 唤醒优化器后峰值 66.4 GB

print(f"四模型 PPO 纯静态底盘 : {m_ppo_static:6.2f} GB/卡 (超 {HBM_MAX} GB 未跑先 OOM!)")
print(f"阶段一 离线 DPO       : 峰值 {m_dpo_peak:4.1f} GB/卡 | 解码  0.00 s | 步时 {t_dpo:5.2f} s")
print(f"阶段二 SGLang GRPO    : 峰值 {m_grpo_peak:4.1f} GB/卡 | 采样缓存 {kv_gqa_gb+gdn_state_gb:.2f} GB | 解码 {t_roll:.2f} s | 步时 {t_roll+t_upd:.2f} s")
