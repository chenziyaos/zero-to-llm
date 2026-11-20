# ZTL-19｜系统｜判断慢在哪，以及请求怎样排队 · 单卡 H100 (25.6 GiB 净池) 案例模型静态双池 vs 2026 统一双向混合池并发上限求解器
POOL_MIB = 26_240                         # 单卡 80GiB H100 扣 54GB 权重与缓冲后净池 26,240 MiB
GDN_SLOT_MIB, BLK_TOKS = 144.0, 16        # 每活跃请求 144 MiB FP32 GDN 状态槽, 分页块长 16 词

def solve_cap(seq_len, kv_kib_tok, unified=True, static_state_share=0.22):
    n_blks = (seq_len + BLK_TOKS - 1) // BLK_TOKS
    kv_mib = n_blks * BLK_TOKS * kv_kib_tok / 1024.0
    if unified:
        b_max = int(POOL_MIB // (GDN_SLOT_MIB + kv_mib))
        dom = "GDN状态槽主导" if GDN_SLOT_MIB > kv_mib else "全局KV页主导"
        return b_max, dom
    b_st = int((POOL_MIB * static_state_share) // GDN_SLOT_MIB)
    b_kv = int((POOL_MIB * (1.0 - static_state_share)) // kv_mib)
    return min(b_st, b_kv), f"min(状态顶{b_st}, KV顶{b_kv})"

for L in [1024, 8192, 32768]:
    s_cap, s_tag = solve_cap(L, 64, unified=False)
    u_cap, u_tag = solve_cap(L, 64, unified=True)
    f_cap, _     = solve_cap(L, 32, unified=True)   # 开启 FP8 KV (32 KiB/词)
    print(f"L={L:5d} | 预划死池(BF16):{s_cap:3d}路[{s_tag}] | "
          f"统一双向池(BF16):{u_cap:3d}路({u_tag}) | 双向池+FP8 KV:{f_cap:3d}路")
