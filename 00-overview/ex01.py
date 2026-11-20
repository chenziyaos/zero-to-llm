# ZTL-00｜总览｜算清四个数：效果、显存、时延与成本 · 零依赖同台核算四款代表模型的显存两栏账
def llm_ledger(name, p_total_b, p_act_b, kv_b_tok, seq_len, batch, cards=1, bytes_w=2, fixed_state_mib=0.0):
    gib, mib = pow(1024, 3), pow(1024, 2)
    peak_tf = 989.0 * cards                       # H100 SXM5 BF16 稠密峰值算力 (TFLOPS)
    m_w_b = p_total_b * 1e9 * bytes_w             # 静态常驻 HBM 权重字节数
    m_kv_b = batch * seq_len * kv_b_tok           # 随序列长度线性增长的全局 KV 字节数
    m_st_b = batch * fixed_state_mib * mib        # 按在线并发扣费的定长状态槽 (GDN / KDA)
    m_tot_gib = (m_w_b + m_kv_b + m_st_b) / gib
    ttft_ms = round((2 * p_act_b * batch * seq_len / 1e3) / peak_tf * 1e3, 1) if p_act_b else "待测"
    return {
        "model": name,
        "w_gib": round(m_w_b / gib, 1),
        "kv_gib": round(m_kv_b / gib, 2),
        "state_gib": round(m_st_b / gib, 2),
        "hbm_gib": f"{m_tot_gib:.1f}/{80*cards}",
        "ttft_floor_ms": ttft_ms,
    }

# 1) 单卡 H100 跑 4 路 64K 并发: 纯 GQA vs 混合架构的显存差异
print(llm_ledger("Qwen3-32B (1xH100, BF16)",    32.0, 32.0, 262144,  65536,  4, cards=1, bytes_w=2))
print(llm_ledger("Qwen3.8-27B (1xH100, BF16)",  27.0, 27.0,  65536,  65536,  4, cards=1, bytes_w=2, fixed_state_mib=144.0))

# 2) 八卡 H100 跑 16 路 64K: MLA 稀疏模型
print(llm_ledger("DeepSeek-V3 (8xH100, FP8)",  671.0, 37.0,  70272,  65536, 16, cards=8, bytes_w=1))

# 3) 64 卡 H100 跑 32 路 256K: 万亿稀疏模型的集群容量
print(llm_ledger("Kimi-K3 (64xH100, FP8)",    2800.0, None,  27648, 262144, 32, cards=64, bytes_w=1, fixed_state_mib=207.0))
