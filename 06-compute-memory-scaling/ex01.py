# ZTL-06｜数学｜一张纸算清算力、显存与缩放 · 一张纸全能账单计算器：一次核清激活算力、16Ψ 最小卡数、256K 两栏缓存、稠密 MFU 与终身算力总账
import math

GIB_TO_GB = (1024 ** 3) / 1e9  # 1 GiB = 1.073741824 GB

def one_page_ledger(p_tot_b, p_act_b, m_lin_b, m_fix_mib, l_glob, d_attn,
                    s_infer=262144, b_infer=1, d_train_t=15.0, d_infer_t=50.0, tps_gpu=0):
    # 1. 第一笔算力账：单词主干 2P_act 与 256K 因果半角全局注意力平均每词算力
    c_tok_gflops = 2.0 * p_act_b
    attn_tok_gflops = (2.0 * l_glob * s_infer * d_attn) / 1e9
    # 2. 第二笔训练显存账：16Ψ 静态底盘（十进制 GB）与物理无溢出理论最小 H100 卡数
    m_static_gb = 16.0 * p_tot_b
    min_h100 = math.ceil(m_static_gb / (80.0 * GIB_TO_GB))
    # 3. 第三笔推理两栏缓存账：B*S*m_linear + B*M_fixed-state（二进制 GiB）
    cache_gib = (b_infer * s_infer * m_lin_b) / (1024 ** 3) + (b_infer * m_fix_mib) / 1024.0
    # 4. 第四笔效率与缩放账：H100 BF16 稠密 MFU、Chinchilla 20P 与全周期总算力（10^24 FLOPs）
    mfu_pct = (6.0 * p_act_b * 1e9 * tps_gpu) / 989e12 * 100.0 if tps_gpu else 0.0
    chinchilla_t = (20.0 * p_act_b) / 1000.0
    c_life_yotta = (6.0 * p_act_b * d_train_t + 2.0 * p_act_b * d_infer_t) / 1000.0
    return {
        "tok_gflops": round(c_tok_gflops, 1),
        "attn_256k_gflops": round(attn_tok_gflops, 1),
        "static_16psi_gb": round(m_static_gb, 1),
        "min_h100": min_h100,
        "cache_256k_gib": round(cache_gib, 2),
        "mfu_pct": round(mfu_pct, 1),
        "chinchilla_20p_T": round(chinchilla_t, 2),
        "c_life_1e24": round(c_life_yotta, 2),
    }

# tps_gpu 是假设的单卡吞吐（按约 45% MFU 反推），只用来演示公式，不是实测
print("Qwen3.8-27B:", one_page_ledger(27.0, 27.0, 65536, 144.0, 16, 6144, tps_gpu=2750))
print("DeepSeek-V3:", one_page_ledger(671.0, 37.0, 70272, 0.0, 61, 16384, d_train_t=14.8, tps_gpu=2000))
print("Kimi-K3:", one_page_ledger(2800.0, 104.0, 27648, 207.0, 24, 12288, tps_gpu=713))
