# ZTL-06｜数学｜一张纸算清算力、显存与缩放 · 三大前沿谱系一张纸对账断言脚本：自动核验单卡 H100 满窗 256K、定长状态反超拐点与压缩缓存
GIB = 1024 ** 3
MIB = 1024 ** 2

# 1. 纯 GQA vs 混合架构容量对账
q3_1m_gib = (64 * 2 * 8 * 128 * 2) * 1_000_000 / GIB
q38_lin_b = 16 * 2 * 4 * 256 * 2                  # 65,536 B = 64.0 KiB/tok
q38_fix_b = 48 * 48 * 128 * 128 * 4               # 150,994,944 B = 144.0 MiB (FP32 定长态)
q38_256k_cache_gib = (q38_lin_b * 262144 + q38_fix_b) / GIB
q38_h100_total_gib = (54e9 / GIB) + q38_256k_cache_gib + 4.0  # 54GB 权重 + 缓存 + 4GiB 运行时

# 2. 万亿 MoE 百万窗口与短窗高并发定长池反超拐点
k3_lin_b = 24 * 576 * 2                           # 27,648 B = 27.0 KiB/tok
k3_fix_b = 69 * 96 * 128 * 128 * 2                # 217,055,232 B = 207.0 MiB (BF16 定长态)
k3_1000k_lin_gib = k3_lin_b * 1_000_000 / GIB     # 100 万词约 25.75 GiB
k3_cross_tokens = k3_fix_b // k3_lin_b            # 7,850 词：短于此值时定长状态池大于 KV 池！

# 3. 稠密 MoE 与后续版本压缩对比
dsv3_lin_kib = (61 * 576 * 2) / 1024              # 68.625 KiB/tok

assert round(q3_1m_gib, 1) == 244.1 and round(q38_256k_cache_gib, 2) == 16.14 and q38_h100_total_gib < 80.0
assert k3_fix_b // MIB == 207 and k3_cross_tokens == 7850
print(round(q38_h100_total_gib, 1), '/ 80 GB')
print('K3 crossover:', k3_cross_tokens)
print(round(q3_1m_gib, 1), '1M |', round(q38_256k_cache_gib, 2), 'cache |', round(q38_h100_total_gib, 1), 'total')
print(round(k3_1000k_lin_gib, 2), '1M |', round(dsv3_lin_kib, 1), 'KiB/tok')
