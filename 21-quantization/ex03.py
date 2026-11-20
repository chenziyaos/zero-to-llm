# ZTL-21｜系统｜用精度换内存要付什么代价 · Qwen3.8-27B 在 RTX 4090 (AWQ) / RTX 5090 (FP8) / H100 (全栈 FP8) 的端云总账核算器
GIB = 1024 ** 3
P_TOTAL = 27.00e9
P_VOCAB = 2 * 248320 * 5120                  # 2.5428B 非绑定头尾词表
P_GATE  = 0.12e9                             # 0.1200B GDN门控与归一化
P_LIN   = P_TOTAL - P_VOCAB - P_GATE
KV_BF16 = 16 * 2 * 4 * 256 * 2               # 16层Gated GQA: 65,536 B (64 KiB/词)
GDN_B   = 48 * 48 * 128 * 128 * 4            # 48层GDN锁死FP32: 144.0 MiB/路 (严禁低比特!)
SEQ_32K = 32768

# 1) 端侧 RTX 4090 (24 GiB, 1.01 TB/s): AWQ W4A16 (g=128) + BF16 KV
w_awq_bytes = P_LIN * ((4.0 + 20.0 / 128.0) / 8.0) + P_VOCAB * 1.0 + P_GATE * 2.0
bw_4090_4k = 1.01e12 / (w_awq_bytes + GDN_B + 4096 * KV_BF16)

# 2) 工作站 RTX 5090 (32 GiB, 1.79 TB/s): 细粒度 FP8 (128x128块) + 校准 FP8 KV
w_fp8_bytes = P_TOTAL * ((8.0 + 32.0 / 16384.0) / 8.0)
free_5090 = (32.0 * GIB - w_fp8_bytes - 2.40 * GIB - GDN_B - SEQ_32K * (KV_BF16 / 2)) / GIB
bw_5090_4k = 1.79e12 / (w_fp8_bytes + GDN_B + 4096 * (KV_BF16 / 2))

# 3) 云端 H100 SXM5 (80 GiB): 32K 满载并发槽扩容 (全 BF16 vs 全栈细粒度 FP8)
req_bf16 = (GDN_B + SEQ_32K * KV_BF16) / GIB
req_fp8  = (GDN_B + SEQ_32K * (KV_BF16 / 2)) / GIB
slots_bf16 = int((80.0 - (P_TOTAL * 2.0 / GIB) - 6.0) // req_bf16)
slots_fp8  = int((80.0 - (w_fp8_bytes / GIB) - 6.0) // req_fp8)

print(f"RTX 4090 AWQ | 权重:{w_awq_bytes/GIB:.2f}GiB | 4K交互解码带宽屋顶:{bw_4090_4k:.1f} tok/s")
print(f"RTX 5090 FP8 | 权重:{w_fp8_bytes/GIB:.2f}GiB | 满跑32K仍余:{free_5090:.2f}GiB | 4K解码屋顶:{bw_5090_4k:.1f} tok/s")
print(f"H100 80G 32K | 全BF16:{slots_bf16}路并发 | 全栈FP8(保GDN FP32):{slots_fp8}路并发 (扩容 {slots_fp8/slots_bf16:.1f}x)")
