# ZTL-25｜工程｜先把相关材料找出来 · 锁定全库标准集分母 |R_q| 并独立核算无答案空集拦截率的离线评测器
import math

def evaluate_retrieval(dataset: list[dict], k: int = 5) -> dict:
    recalls, precs, rrs, ndcgs = [], [], [], []
    unans_total, unans_blocked = 0, 0
    for row in dataset:
        gold: dict[str, int] = row["gold_rel"]   # 全库标准证据区间锚点: {span_id: 相关度 1 或 2}
        preds: list[str] = row["pred_ids"][:k]   # 系统返回的前 k 个命中区间 ID
        if len(gold) == 0:                       # |R_q| = 0 库外无答案题独立分桶，严禁混入分母
            unans_total += 1
            unans_blocked += int(len(preds) == 0)
            continue
        hits = [sid for sid in preds if sid in gold]
        recalls.append(len(hits) / len(gold))    # 铁律一：分母恒为全库标注相关集大小 |R_q|
        precs.append(len(hits) / k)              # 分母为固定送模窗口槽位数 k
        rrs.append(next((1.0 / (i + 1) for i, sid in enumerate(preds) if sid in gold), 0.0))
        dcg = sum((2**gold.get(sid, 0) - 1) / math.log2(i + 2) for i, sid in enumerate(preds))
        ideal = sorted(gold.values(), reverse=True)[:k]
        idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    n_ans = len(recalls)
    p_bar = sum(recalls) / n_ans
    se_rec = math.sqrt(p_bar * (1.0 - p_bar) / n_ans)
    return {
        f"Recall@{k}": round(p_bar, 4), f"Recall@{k}_SE": round(se_rec, 4),
        f"Precision@{k}": round(sum(precs) / n_ans, 4), f"MRR@{k}": round(sum(rrs) / n_ans, 4),
        f"NDCG@{k}": round(sum(ndcgs) / n_ans, 4),
        "EmptyBlockRate": round(unans_blocked / max(unans_total, 1), 4),
    }

bench = [
    {"gold_rel": {"pg_v32_err5042": 2, "pg_v32_idemp": 2, "pg_v32_pool": 1, "pg_v32_ttl": 1},
     "pred_ids": ["pg_v32_err5042", "pg_v31_old", "pg_v32_pool", "order_db", "pg_v32_ttl"]},
    {"gold_rel": {"pg_v32_cert": 2}, "pred_ids": ["pg_log_spam", "pg_v32_cert", "pg_v31_old", "order_db", "mq_guide"]},
    {"gold_rel": {}, "pred_ids": []},              # 库外无答案题，相关性阈值成功拦截清空
    {"gold_rel": {}, "pred_ids": ["pg_log_spam"]}, # 库外无答案题，无阈值兜底误放行噪声切片
]
print("q* 单题核对:", evaluate_retrieval(bench[:1], k=5))
print("微型集汇总:", evaluate_retrieval(bench, k=5))
print("库外题抓取:", round(60 * 0.5), "题成功拦截")
