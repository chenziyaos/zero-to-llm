# ZTL-26｜工程｜让回答有可追溯的依据 · U 型首尾重排 + 不可变 XML 证据信封组装 + 短角标确定性字节区间与哈希反查
import hashlib

RANKED_CHUNKS = [
    {"doc_id": "pg_v32_timeout", "ver": "v3.2", "b_start": 4120,
     "text": "ERR-5042 清算读超时：建议 max_retry=3, backoff=200ms，严禁关闭幂等锁。"},
    {"doc_id": "pg_v32_idemp",   "ver": "v3.2", "b_start": 8300,
     "text": "重试请求须携带唯一 Idempotency-Key，服务端锁窗口为 24h。"},
    {"doc_id": "pg_v32_pool",    "ver": "v3.2", "b_start": 6200,
     "text": "高并发清算期建议连接池上限 max_pool_size=200，空闲 60s 回收。"},
    {"doc_id": "pg_v31_legacy",  "ver": "v3.1", "b_start": 1050,
     "text": "旧版 v3.1 曾设 read_timeout=5000ms，该配置已在 v3.2 废止。"},
    {"doc_id": "pg_v32_metrics", "ver": "v3.2", "b_start": 9410,
     "text": "超时告警指标为 gw_upstream_timeout_total，阈值 P99 < 300ms。"},
]

def build_u_envelope(ranked: list[dict]) -> tuple[str, dict[int, dict]]:
    u_ordered = ranked[0::2] + ranked[1::2][::-1]  # [Rank1, Rank3, Rank5, Rank4, Rank2]
    xml_blocks, registry = [], {}
    for idx, ch in enumerate(u_ordered, start=1):
        raw = ch["text"].encode("utf-8")
        sha12 = hashlib.sha256(raw).hexdigest()[:12]
        span = f"{ch['b_start']}:{ch['b_start'] + len(raw)}"
        registry[idx] = {**ch, "byte_span": (ch["b_start"], ch["b_start"] + len(raw)), "sha12": sha12}
        xml_blocks.append(
            f'<doc id="{idx}" doc_id="{ch["doc_id"]}" version="{ch["ver"]}" '
            f'span="{span}" sha256="{sha12}">\n{ch["text"]}\n</doc>'
        )
    return "\n".join(xml_blocks), registry

def resolve_citation(cite_id: int, snippet: str, registry: dict[int, dict]) -> dict:
    meta = registry[cite_id]
    raw_doc = meta["text"].encode("utf-8")
    assert hashlib.sha256(raw_doc).hexdigest()[:12] == meta["sha12"], "源文档快照已漂移!"
    sub = snippet.encode("utf-8")
    b0 = meta["byte_span"][0] + max(raw_doc.find(sub), 0)
    return {"doc_id": meta["doc_id"], "ver": meta["ver"], "exact_bytes": (b0, b0 + len(sub)), "sha12": meta["sha12"]}

xml_ctx, reg = build_u_envelope(RANKED_CHUNKS)
print("U型槽位顺序:", [f"[{i}]{reg[i]['doc_id']}({reg[i]['ver']})" for i in reg])
print("反查角标[5]:", resolve_citation(5, "Idempotency-Key", reg))
print("信封总长:", len(xml_ctx), "字符 | 注册角标数:", len(reg))
