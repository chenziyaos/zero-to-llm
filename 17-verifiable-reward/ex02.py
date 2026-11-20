# ZTL-17｜后训练｜先验证奖励再训练 · 融合 DAPO 零方差死题过滤、Clip-Higher(0.20/0.28) 非对称裁剪、词元级归约与 k3 KL 的 GRPO 更新步
import math

def dapo_grpo_step(group_rewards, lp_new, lp_old, lp_ref, tok_entropies, eps_lo=0.20, eps_hi=0.28, beta=0.04):
    kept, adv_groups = [], []
    for gid, rs in enumerate(group_rewards):
        acc = sum(1 for r in rs if r >= 1.0) / len(rs)
        if not (0.0 < acc < 1.0):  # DAPO 动态过滤：当场剔除全错(acc=0)与全对(acc=1)零方差死题
            continue
        mu = sum(rs) / len(rs)
        std = math.sqrt(sum((r - mu) ** 2 for r in rs) / len(rs))
        kept.append(gid)
        adv_groups.append([(r - mu) / (std + 1e-6) for r in rs])

    surr_sum, kl_sum, ent_sum, n_tok, n_clip_hi = 0.0, 0.0, 0.0, 0, 0
    for pos, gid in enumerate(kept):
        for i, adv in enumerate(adv_groups[pos]):
            for ln, lo, lr, ent in zip(lp_new[gid][i], lp_old[gid][i], lp_ref[gid][i], tok_entropies[gid][i]):
                ratio = math.exp(ln - lo)
                r_clip = min(max(ratio, 1.0 - eps_lo), 1.0 + eps_hi)  # Clip-Higher 放宽上界至 1.28
                if adv > 0 and ratio > 1.0 + eps_hi:
                    n_clip_hi += 1
                delta = lr - ln
                kl_k3 = math.exp(delta) - delta - 1.0                 # Schulman k3 恒非负无偏 KL
                surr_sum += min(ratio * adv, r_clip * adv)
                kl_sum += kl_k3; ent_sum += ent; n_tok += 1
    loss = -(surr_sum / n_tok) + beta * (kl_sum / n_tok)              # 按有效批总词元数统一归约
    return len(kept), loss, kl_sum / n_tok, ent_sum / n_tok, n_clip_hi

# 5 道候选提示组(G=4)：前 3 组为全对/全错零方差死题(占 60%)，后 2 组为具备对比梯度的有效组
rewards_5g = [[1.0]*4, [0.0]*4, [1.0]*4, [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]
lp_o = [[[-2.30, -0.80] for _ in range(4)] for _ in range(5)]
lp_n = [[[-2.04, -0.74] if i == 0 else [-2.42, -0.86] for i in range(4)] for _ in range(5)]
lp_r = [[[-2.35, -0.82] for _ in range(4)] for _ in range(5)]
ents = [[[0.42, 0.36] for _ in range(4)] for _ in range(5)]

kept, loss, kl_m, ent_m, hi_c = dapo_grpo_step(rewards_5g, lp_n, lp_o, lp_r, ents)
print("== DAPO GRPO 单步更新 ==")
print(f"有效组:{kept}/5 | 词元级损失:{loss:.4f}")
print(f"k3-KL:{kl_m:.4f} | 策略熵:{ent_m:.2f} nats | 上界裁剪词:{hi_c}")
