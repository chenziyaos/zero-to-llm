# ZTL-09｜模型｜让模型知道顺序 · 零依赖 Python 脚本一键对账 DeepSeek-V4 (64K->1M)、Qwen3.8-27B (262K->1M) 与 Qwen3-32B 的分频谱及 KV 显存
import math

def audit_1m_rope(name, d_rot, base, orig_max, target_max, kv_kib_per_tok, fixed_state_mib=0.0):
    s = target_max / orig_max
    pairs = d_rot // 2
    thetas = [base ** (-2.0 * j / d_rot) for j in range(pairs)]
    wls = [2.0 * math.pi / th for th in thetas]
    rots = [orig_max / wl for wl in wls]                        # 原生窗口内旋转圈数 r_j
    gammas = [min(1.0, max(0.0, (r - 1.0) / 31.0)) for r in rots]
    th_yarn = [(1.0 - g) * (th / s) + g * th for g, th in zip(gammas, thetas)]
    wl_yarn_max = 2.0 * math.pi / th_yarn[-1]                   # YaRN 拉伸后的最慢平面波长
    nh = sum(1 for g in gammas if g == 1.0)                     # 高频无损保留对数 (gamma=1)
    nr = sum(1 for g in gammas if 0.0 < g < 1.0)                # 中频平滑斜坡对数
    nl = sum(1 for g in gammas if g == 0.0)                     # 低频 1/s 线性内插对数 (gamma=0)
    mscale = 0.1 * math.log(s) + 1.0                            # cos/sin 预乘幅度补偿因子
    kv_bf16_gib = kv_kib_per_tok * target_max / (1024 ** 2)     # 满窗单序列线性增长 BF16 KV
    print(f"{name:24s} | s={s:4.1f}x | 高/斜/低={nh:2d}/{nr:2d}/{nl:2d} | "
          f"最慢波长->{wl_yarn_max/1e3:9.1f}K | mscale={mscale:.4f} (1/t={mscale**2:.4f}) | "
          f"满窗BF16 KV={kv_bf16_gib:5.1f}GiB (+定长{fixed_state_mib:.0f}MiB)")

if __name__ == "__main__":
    audit_1m_rope("DeepSeek-V4 (64K->1M)",   64,  10_000.0,     65536,  1048576, 68.625, 0.0)
    audit_1m_rope("Qwen3.8-27B (262K->1M)",  64,  10_000_000.0, 262144, 1048576, 64.0,   72.0)
    audit_1m_rope("Qwen3-32B   (32K->128K)", 128, 1_000_000.0,  32768,  131072,  256.0,  0.0)
