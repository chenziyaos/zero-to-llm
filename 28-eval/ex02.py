# ZTL-28｜工程｜先写评测再改系统 · 零外部依赖的 Paired Bootstrap 置信区间与精确二项 McNemar 上线门禁脚本
import math
import random

def exact_mcnemar_p(n01: int, n10: int) -> float:
    """零外部依赖计算精确双侧二项 McNemar p 值：2 * sum_{k=0}^{min} C(m, k) * 0.5^m"""
    m = n01 + n10
    if m == 0:
        return 1.0
    k_max = min(n01, n10)
    tail = sum(math.comb(m, k) for k in range(k_max + 1)) * (0.5 ** m)
    return min(1.0, 2.0 * tail)

def paired_eval_gate(scores_a: list[float], scores_b: list[float], n_boot: int = 10000, seed: int = 2026) -> dict:
    """支持 {0,1} 二值或 avg@k 连续实数得分的同题配对统计门禁"""
    n = len(scores_a)
    diffs = [b - a for a, b in zip(scores_a, scores_b)]
    mean_d = sum(diffs) / n
    # 独立双样本标准误 vs 同题配对解析标准误
    pa, pb = sum(scores_a) / n, sum(scores_b) / n
    se_ind = math.sqrt((pa * (1.0 - pa) + pb * (1.0 - pb)) / n)
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    se_paired = math.sqrt(var_d / n)
    # 0/1 分歧统计与精确 McNemar 检验
    n01 = sum(1 for a, b in zip(scores_a, scores_b) if a < 0.5 and b >= 0.5)
    n10 = sum(1 for a, b in zip(scores_a, scores_b) if a >= 0.5 and b < 0.5)
    z_mc = (n01 - n10) / math.sqrt(n01 + n10) if (n01 + n10) > 0 else 0.0
    p_val = exact_mcnemar_p(n01, n10)
    # Paired Bootstrap：用同一组随机题号同步重抽样差值向量
    rng = random.Random(seed)
    boot_means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    ci_lo, ci_hi = boot_means[int(0.025 * n_boot)], boot_means[int(0.975 * n_boot)]
    return {
        "delta_pt": round(mean_d * 100, 2), "se_ind_pt": round(se_ind * 100, 2),
        "se_paired_pt": round(se_paired * 100, 2), "z_mcnemar": round(z_mc, 2),
        "p_exact": round(p_val, 4), "ci95_pt": (round(ci_lo * 100, 2), round(ci_hi * 100, 2)),
    }

# 1. 验证本章 54 题分歧例（39 胜 15 负，净赢 +4.0pt）与过渡版 B2-alpha（30 胜 21 负，+1.5pt）
sa = [1.0] * 454 + [0.0] * 92 + [0.0] * 39 + [1.0] * 15
sb_54 = [1.0] * 454 + [0.0] * 92 + [1.0] * 39 + [0.0] * 15
sb_alpha = [1.0] * 448 + [0.0] * 101 + [1.0] * 30 + [0.0] * 21
sa_alpha = [1.0] * 448 + [0.0] * 101 + [0.0] * 30 + [1.0] * 21
print("54题分歧例:", paired_eval_gate(sa, sb_54))
print("B2-alpha待定:", paired_eval_gate(sa_alpha, sb_alpha))
print("统计对比：54题分歧 p=0.0015 过关，51题分歧 p=0.2624 未过关")
