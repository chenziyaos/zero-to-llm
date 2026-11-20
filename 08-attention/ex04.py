# ZTL-08｜模型｜把注意力写成矩阵 · 零依赖纯 Python 总账核验：一键复算 2020→2026 六代注意力每词缓存
# 统一按半精度 2 B/elem 核算线性增长 KV 与每请求定长状态池
gpt3_b      = 2 * 96 * 96 * 128 * 2          # 4,718,592 B = 4.50 MiB/tok (2020 初代 MHA)
qwen3_32b_b = 2 * 64 * 8 * 128 * 2           #   262,144 B = 256.0 KiB/tok (2025 纯 GQA 8:1)
dsv3_b      = 61 * (512 + 64) * 2            #    70,272 B =  68.6 KiB/tok (2024 MLA 潜空间吸收)
qwen38_b    = 2 * 16 * 4 * 256 * 2           #    65,536 B =  64.0 KiB/tok (2026 3:1 GDN + Gated GQA)
kimik3_b    = 24 * (512 + 64) * 2            #    27,648 B =  27.0 KiB/tok (2026 3:1 KDA + Gated MLA)
# DeepSeek-V4 块压缩的每词字节数依赖实现细节，这里只做架构层说明

gdn_bf16_mib = (48 * 48 * 128 * 128 * 2) / (1024 ** 2)   # 72.0 MiB (FP32 态为 144.0 MiB)
kda_bf16_mib = (69 * 96 * 128 * 128 * 2) / (1024 ** 2)   # 207.0 MiB
compress_ratio_to_kimi = gpt3_b / kimik3_b               # 170.7x (GPT-3 → Kimi K3)

assert qwen3_32b_b == 4 * qwen38_b and gdn_bf16_mib == 72.0 and kda_bf16_mib == 207.0
print(f"天梯(per-tok): GPT-3={gpt3_b/1048576:.2f}MiB -> Qwen3={qwen3_32b_b/1024:.0f}KiB -> DS-V3={dsv3_b/1024:.1f}KiB")
print(f"2026 代表架构  : Qwen3.8={qwen38_b/1024:.0f}KiB(+{gdn_bf16_mib:.0f}MiB) | Kimi-K3={kimik3_b/1024:.0f}KiB(+{kda_bf16_mib:.0f}MiB)")
print(f"压缩倍数与1M词显存: GPT-3→Kimi-K3 {compress_ratio_to_kimi:.1f}倍 | 1M词 KV: Qwen3-32B={qwen3_32b_b*1048576/1073741824:.2f}GiB vs Kimi-K3={kimik3_b*1048576/1073741824:.2f}GiB")
