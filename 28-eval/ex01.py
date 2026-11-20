# ZTL-28｜工程｜先写评测再改系统 · 双盲位置交换消偏统计与 Rogan-Gladen 真实通过率及 95% 置信区间校准脚本
import math

def resolve_position_swap(ab_votes: list[str], ba_votes: list[str]) -> dict:
    """输入两轮已映射回真实身份 ('A'/'B'/'T') 的盲评票，不一致矛盾票强制归入 bias_tie"""
    tally = {"B_win": 0, "A_win": 0, "clean_tie": 0, "bias_tie": 0}
    for v1, v2 in zip(ab_votes, ba_votes):
        if v1 == "B" and v2 == "B":
            tally["B_win"] += 1
        elif v1 == "A" and v2 == "A":
            tally["A_win"] += 1
        elif v1 == "T" and v2 == "T":
            tally["clean_tie"] += 1
        else:
            tally["bias_tie"] += 1
    n = len(ab_votes)
    # 双向对称净胜率：赢计 1 分、平局计 0.5 分
    b_score = (tally["B_win"] + 0.5 * (tally["clean_tie"] + tally["bias_tie"])) / n
    return {**tally, "B_debiased_winrate": round(b_score, 4)}

def calibrate_rogan_gladen(p_raw: float, n_eval: int, tp: int, m_pos: int, tn: int, m_neg: int):
    """结合人工金标 (tp/m_pos, tn/m_neg) 反解去偏真实达标率与 Delta 方法 95% CI"""
    q1, q0 = tp / m_pos, tn / m_neg
    youden = q1 + q0 - 1.0
    if youden <= 0.0:
        raise ValueError("裁判约登指数 <= 0，丧失正向判别力")
    theta = min(max((p_raw + q0 - 1.0) / youden, 0.0), 1.0)
    var_p = p_raw * (1.0 - p_raw) / n_eval
    var_q1 = (theta ** 2) * q1 * (1.0 - q1) / m_pos
    var_q0 = ((1.0 - theta) ** 2) * q0 * (1.0 - q0) / m_neg
    se = math.sqrt(var_p + var_q1 + var_q0) / youden
    return round(theta, 4), round(1.96 * se, 4)

# 1. 复现 300 道开放题正反翻转 11.0pt 现场：首轮 B 占首槽赢 153 题，次轮掉至末槽仅保住 120 题
r_ab = ["B"] * 153 + ["A"] * 120 + ["T"] * 27   # 单向首轮 B 表观胜率 (153 + 13.5)/300 = 55.5%
r_ba = ["B"] * 120 + ["A"] * 33 + ["A"] * 120 + ["T"] * 27  # 次轮对调后 B 胜率跌至 44.5%
print("双盲换位裁决:", resolve_position_swap(r_ab, r_ba))

# 2. 验证手算例与排障网关 120 题金标校准
print("q0=0.75 报 82% ->", calibrate_rogan_gladen(0.82, 600, tp=76, m_pos=80, tn=30, m_neg=40))
print("q0=0.65 报 80% ->", calibrate_rogan_gladen(0.80, 600, tp=76, m_pos=80, tn=26, m_neg=40))
print("原子清单实测校准 ->", calibrate_rogan_gladen(0.768, 600, tp=77, m_pos=80, tn=36, m_neg=40))
