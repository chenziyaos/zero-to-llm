# ZTL-26｜工程｜让回答有可追溯的依据 · 两阶段引文与原子断言忠实度核验器：确定性锚点校验 + NLI 蕴含门禁与原文回退
import re

DOCS = {
    1: {"ver": "v3.2", "text": "ERR-5042 清算读超时：建议 max_retry=3, backoff=200ms，严禁关闭幂等锁。"},
    2: {"ver": "v3.2", "text": "高并发清算期建议连接池上限 max_pool_size=200，空闲 60s 回收。"},
    5: {"ver": "v3.2", "text": "重试请求须携带唯一 Idempotency-Key，服务端锁窗口为 24h。"},
}

def extract_anchors(text: str) -> set[str]:
    return set(re.findall(r"ERR-\d+|\d+(?:ms|s|h)?", text))

def minicheck_entail_prob(premise: str, claim: str) -> float:
    # 生产环境替换为 MiniCheck-0.5B GPU 单批前向: model.score(premises, claims)
    if "重试上限" in claim and "200" in claim and "max_pool_size=200" in premise:
        return 0.04  # 戳穿角标洗白：原文 200 修饰连接池而非重试上限！
    if "max_retry=3" in claim and "max_retry=3" in premise:
        return 0.96
    if "Idempotency-Key" in claim and "Idempotency-Key" in premise:
        return 0.94
    return 0.15

def two_stage_verify(claims: list[dict], docs: dict[int, dict], gamma: float = 0.75) -> dict:
    passed, safe_lines = 0, []
    for item in claims:
        c, cids = item["claim"], item["cites"]
        valid_ids = [j for j in cids if j in docs]
        joined = " ".join(docs[j]["text"] for j in valid_ids)
        # 阶段一：角标未越界且断言内错误码/数值锚点全部在所引原文中出现
        anchors_ok = bool(valid_ids) and (extract_anchors(c) <= extract_anchors(joined))
        # 阶段二：核算 max_{j in cite(c)} P_entail(d_j => c) 并裁剪凑数冗余角标
        scores = {j: minicheck_entail_prob(docs[j]["text"], c) for j in valid_ids} if anchors_ok else {}
        p_max = max(scores.values(), default=0.0)
        if anchors_ok and p_max >= gamma:
            passed += 1
            pruned = [j for j, s in scores.items() if s >= gamma]
            safe_lines.append(f"{c} {pruned}")
        else:
            raw_quote = docs[valid_ids[0]]["text"] if valid_ids else "引用编号越界，已拦截"
            safe_lines.append(f"[已回退原文摘录] {raw_quote}")
    return {"S_faith": round(passed / len(claims), 3), "safe_output": safe_lines}

ans_claims = [
    {"claim": "遇到 ERR-5042 建议设 max_retry=3 且退避 200ms", "cites": [1, 2]},
    {"claim": "高并发期可将重试上限调至 200", "cites": [2]},
    {"claim": "重试须携带 Idempotency-Key 且锁窗口 24h", "cites": [5]},
]
res = two_stage_verify(ans_claims, DOCS, gamma=0.75)
print("核验前原始忠实度:", res["S_faith"])
for ln in res["safe_output"]:
    print(" ->", ln)
