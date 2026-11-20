# ZTL-19｜系统｜判断慢在哪，以及请求怎样排队 · 单卡 H100 上案例模型批大小撬动线性层吞吐与私有缓存带宽硬墙模拟器
F_PEAK, BW_HBM = 989e12, 3.35e12          # H100 SXM5: 989 TFLOPS, 3.35 TB/s
P_W, D, L_TOT = 27.0e9, 5120, 64          # Qwen3.8-27B: 27B 稠密 (BF16 权重 54.0 GB)
GQA_G, B_KV, KV_TOK = 6, 2, 65_536        # 16层 Gated GQA (24Q/4KV, g=6), 64 KiB/词
GDN_STATE_B = 48 * 48 * 128 * 128 * 4     # 48层 GDN FP32 定长状态: 144.0 MiB (150,994,944 B)
SEQ_LEN = 8192                            # 假设活跃请求当前上下文均为 8,192 词

for B in [1, 8, 32, 64, 128, 256, 512]:
    ai_lin = (2 * B * D * D) / (2 * (D * D + 2 * B * D))
    t_lin = max((2 * B * P_W) / F_PEAK, (2 * P_W + 4 * B * D * L_TOT) / BW_HBM)
    bytes_indep = B * (SEQ_LEN * KV_TOK + 2 * GDN_STATE_B)  # 独立KV读 + 独立GDN状态读写
    t_cache = bytes_indep / BW_HBM
    tpot_ms = (t_lin + t_cache) * 1e3
    mem_gib = B * (SEQ_LEN * KV_TOK + GDN_STATE_B) / (1024 ** 3)
    print(f"B={B:3d} | AI_lin={ai_lin:5.1f} | t_lin={t_lin*1e3:5.2f}ms | "
          f"t_cache={t_cache*1e3:5.2f}ms | TPOT={tpot_ms:5.2f}ms | "
          f"TPS={B/(t_lin+t_cache):4.0f} tok/s | 池需={mem_gib:5.1f}GiB")
