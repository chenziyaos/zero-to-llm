# ZTL-25｜工程｜先把相关材料找出来 · 网关排障工单 q* 上的 BM25 代码同构倒排 + 归一化向量 + RRF(κ=60) 合流验证
import math
import numpy as np

# 网关知识库 V1 增强切片（已焊入 PaymentGateway v3.2 路径头；完整回答 q* 需集齐 err5042 与 idemp）
CORPUS = [
    ("pg_v32_err5042", ["paymentgateway", "v3.2", "err-5042", "清算通道", "读超时", "max_retry", "backoff"]),
    ("pg_v32_err5041", ["paymentgateway", "v3.2", "err-5041", "商户公钥", "签名失败", "核对证书"]),
    ("pg_v32_pool",    ["paymentgateway", "v3.2", "高并发", "连接池", "指数退避", "空闲重置"]),
    ("order_db",       ["orderservice", "mysql", "主从延迟", "只读实例", "降级开关"]),
    ("pg_v32_idemp",   ["paymentgateway", "v3.2", "idempotency-key", "幂等规范", "网络重试", "防止重复扣费"]),
]
EMB = {  # 模拟双塔向量：难区分 5042/5041(err5042 掉到第3)，但精准对齐“怕重试多扣钱 <-> 防止重复扣费”(idemp 排第1)
    "pg_v32_err5042": [0.60, 0.52, 0.32, 0.05], "pg_v32_err5041": [0.65, 0.58, 0.30, 0.05],
    "pg_v32_pool":    [0.76, 0.22, 0.45, 0.08], "order_db":       [0.10, 0.08, 0.05, 0.96],
    "pg_v32_idemp":   [0.56, 0.15, 0.88, 0.04],
}

def bm25_search(q_tokens: list[str], k1: float = 1.2, b: float = 0.75) -> list[str]:
    avgdl = sum(len(toks) for _, toks in CORPUS) / len(CORPUS)
    hits = {}
    for cid, toks in CORPUS:
        score = 0.0
        for w in q_tokens:
            df = sum(1 for _, dt in CORPUS if w in dt)
            if df == 0 or w not in toks:
                continue
            idf = math.log(1.0 + (len(CORPUS) - df + 0.5) / (df + 0.5))
            tf = toks.count(w)
            score += idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * len(toks) / avgdl))
        if score > 0.0:
            hits[cid] = score
    return [cid for cid, _ in sorted(hits.items(), key=lambda x: -x[1])]

def dense_search(q_vec: list[float]) -> list[str]:
    qv = np.asarray(q_vec, dtype=np.float64) / np.linalg.norm(q_vec)
    sims = {cid: float(qv @ (np.asarray(v) / np.linalg.norm(v))) for cid, v in EMB.items()}
    return [cid for cid, _ in sorted(sims.items(), key=lambda x: -x[1])]

def rrf_fuse(rankings: list[list[str]], kappa: int = 60) -> list[tuple[str, float]]:
    fused = {}
    for r_list in rankings:
        for rank_idx, cid in enumerate(r_list, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (kappa + rank_idx)
    return sorted(fused.items(), key=lambda x: -x[1])

lex_res = bm25_search(["paymentgateway", "v3.2", "err-5042", "读超时", "怕重试多扣钱"])
vec_res = dense_search([0.60, 0.35, 0.72, 0.02])
rrf_res = rrf_fuse([lex_res, vec_res], kappa=60)
print("BM25 排序:", lex_res, "| Dense 前3:", vec_res[:3])
print("RRF Top-3:", [(c, round(s, 5)) for c, s in rrf_res[:3]])
print("双路共识:", rrf_res[0][0], "与", rrf_res[1][0], "稳居前二")
