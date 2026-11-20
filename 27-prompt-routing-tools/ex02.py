# ZTL-27｜工程｜提示、路由与工具边界 · MCP 工具安全执行网关：身份覆写、参数越界拦截、W1 幂等去重与 W2 diff 审批状态机
import hashlib, json

MCP_TOOLS = {
    "query_err_metrics": {"tier": "R0", "fields": {"err_code", "window_min"}},
    "set_canary_retry":  {"tier": "W1", "fields": {"service", "retry_max"}, "bounds": ("retry_max", 0, 3)},
    "update_ratelimit_whitelist": {"tier": "W2", "fields": {"merchant_id", "new_qps"}, "bounds": ("new_qps", 10, 50000)},
}
IDEM_CACHE, PENDING_DIFFS = {}, {}

def mcp_safety_gateway(sess: dict, call_slot: str, tool: str, raw_args: dict, approve_sig: str = "") -> dict:
    if tool not in MCP_TOOLS:
        return {"ok": False, "code": "TOOL_NOT_ALLOWED", "recoverable": False}
    spec = MCP_TOOLS[tool]
    # 1. 无条件剥离模型伪造的身份字段，强制绑定服务端验签登录态
    args = {k: v for k, v in raw_args.items() if k not in {"operator_id", "tenant_id"}}
    if set(args) != spec["fields"]:
        return {"ok": False, "code": "SCHEMA_MISMATCH", "recoverable": True, "hint": f"仅允许字段 {sorted(spec['fields'])}"}
    if "bounds" in spec:
        bk, lo, hi = spec["bounds"]
        if not (lo <= int(args[bk]) <= hi):
            return {"ok": False, "code": "OUT_OF_BOUNDS", "recoverable": True, "hint": f"{bk} 须在 [{lo}, {hi}] 内"}
    scoped = {**args, "tenant_id": sess["tenant_id"], "operator_id": sess["operator_id"]}
    if spec["tier"] == "R0":
        return {"ok": True, "tier": "R0", "status": "EXECUTED", "scoped": scoped}
    # 2. W1/W2 统一计算确定性幂等键 Idempotency-Key，防超时重试重复变更
    idem_seed = f"{sess['session_id']}:{call_slot}:{tool}:{json.dumps(scoped, sort_keys=True)}"
    idem_key = hashlib.sha256(idem_seed.encode()).hexdigest()[:16]
    if idem_key in IDEM_CACHE:
        return {**IDEM_CACHE[idem_key], "idempotent_replay": True}
    # 3. W2 高危写操作：原地挂起并生成变更前后 diff 卡片，无人工签名绝不执行
    if spec["tier"] == "W2" and PENDING_DIFFS.get(idem_key) != approve_sig:
        PENDING_DIFFS[idem_key] = f"APPROVE-{idem_key[:6]}"
        diff_card = {"target": args["merchant_id"], "before_qps": 2000, "after_qps": args["new_qps"]}
        return {"ok": False, "tier": "W2", "status": "PENDING_APPROVAL", "idem_key": idem_key, "diff": diff_card}
    res = {"ok": True, "tier": spec["tier"], "status": "EXECUTED", "idem_key": idem_key, "scoped": scoped}
    IDEM_CACHE[idem_key] = res
    return res

sess = {"session_id": "INC-20261117", "tenant_id": "PAY_CORE", "operator_id": "SRE_LIN"}
w1_a = mcp_safety_gateway(sess, "slot_1", "set_canary_retry", {"service": "pay-gw", "retry_max": 2, "tenant_id": "HACK"})
w1_b = mcp_safety_gateway(sess, "slot_1", "set_canary_retry", {"service": "pay-gw", "retry_max": 2})
w2_p = mcp_safety_gateway(sess, "slot_2", "update_ratelimit_whitelist", {"merchant_id": "M_9981", "new_qps": 0})
w2_s = mcp_safety_gateway(sess, "slot_2", "update_ratelimit_whitelist", {"merchant_id": "M_9981", "new_qps": 500})
w2_ok = mcp_safety_gateway(sess, "slot_2", "update_ratelimit_whitelist", {"merchant_id": "M_9981", "new_qps": 500}, PENDING_DIFFS[w2_s["idem_key"]])
print("1.W1防伪造与幂等:", w1_a["scoped"]["tenant_id"], "| 超时重试重放=", w1_b.get("idempotent_replay"))
print("2.参数越界拦截 :", w2_p["code"], w2_p["hint"])
print("3.W2挂起与审批 :", w2_s["status"], w2_s["diff"], "-> 签批后:", w2_ok["status"])
