# ZTL-24｜系统｜长上下文与混合专家怎样改变账单 · 稠密混合 Qwen3.8-27B 与细粒度 MoE 活跃专家覆盖率及解码权重算术强度模拟
import random

def sim_moe_reuse(name, psi_tot, psi_act, e_num, k_sel, batches, trials=400):
    rng = random.Random(2026)
    psi_routed = 0.0 if e_num == 1 else (psi_tot - psi_act) / (1.0 - k_sel / e_num)
    psi_shared = psi_tot - psi_routed
    print(f"=== {name} (总参={psi_tot:g}B, 激活={psi_act:g}B, 常驻={psi_shared:.2f}B, {e_num}选{k_sel}) ===")
    for b in batches:
        p_cov = 1.0 if e_num == 1 else 1.0 - (1.0 - k_sel / e_num) ** b
        sim_cov = 1.0 if e_num == 1 else sum(
            len({e for _ in range(b) for e in rng.sample(range(e_num), k_sel)})
            for _ in range(trials)
        ) / (trials * e_num)
        w_gb = psi_shared + p_cov * psi_routed  # FP8 口径: 1 Byte/param
        ai = (2.0 * b * psi_act) / w_gb
        print(f"B={b:4d} | 覆盖={100*p_cov:5.1f}% (模拟={100*sim_cov:5.1f}%) | "
              f"读权重={w_gb:6.1f} GB | 算术强度={ai:6.1f} F/B")

sim_moe_reuse("Qwen3.8-27B",   27.0, 27.0,   1, 1, [1, 64, 1024])
sim_moe_reuse("DeepSeek-V3",  671.0, 37.0, 256, 8, [1, 16, 64, 128, 1024])
sim_moe_reuse("Kimi K2",     1000.0, 32.0, 384, 8, [1, 16, 64, 128, 1024])
