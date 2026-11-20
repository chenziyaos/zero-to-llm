# ZTL-30｜交付｜交一个可复查的项目 · 金丝雀灰度无真值代理指标监控与四元耦合包（Coupled Bundle）原子回滚控制器
import math

BUNDLES = {
    "bundle_v19": {"model": "qwen3.8-27b-fp8@e10a", "prompt": "p_v19", "index": "kb_pay_v19", "embed": "bge-m3-v1.1", "router": "r_v19"},
    "bundle_v20": {"model": "qwen3.8-27b-fp8@e3b0", "prompt": "p_v20", "index": "kb_pay_v20", "embed": "bge-m3-v1.2", "router": "r_v20"},
}
INDEX_EMBED = {"kb_pay_v19": "bge-m3-v1.1", "kb_pay_v20": "bge-m3-v1.2"}

def min_canary_n(conf: float = 0.95, p_def: float = 0.01) -> int:
    return math.ceil(math.log(1.0 - conf) / math.log(1.0 - p_def))

def canary_guard_and_rollback(active_ptr: str, base: dict, obs: dict) -> tuple[str, str, list[str]]:
    alerts = []
    if obs["ttft_p99_ms"] > 800 or obs["tpot_p99_ms"] > 25.0:
        alerts.append(f"LATENCY_BREACH:{obs['ttft_p99_ms']}ms/{obs['tpot_p99_ms']}ms")
    if obs["tool_err_rate"] > 0.02:
        alerts.append(f"MCP_TOOL_ERR:{obs['tool_err_rate']:.1%}")
    if obs["nli_low_rate"] > 0.05 or obs["downvote_rate"] > base["downvote_rate"] + 0.03:
        alerts.append(f"FAITH_OR_UX_DROP:{obs['nli_low_rate']:.1%}/{obs['downvote_rate']:.1%}")
    n1, n2, p1, p2 = base["n"], obs["n"], base["empty_rate"], obs["empty_rate"]
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(max(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2), 1e-12))
    z_empty = (p2 - p1) / se
    if n2 >= 100 and abs(z_empty) > 3.0:
        alerts.append(f"EMPTY_RATE_SPIKE:z={z_empty:.2f}({p2:.1%})")
    if alerts:
        rb_ptr = "bundle_v19"
        bd = BUNDLES[rb_ptr]
        assert INDEX_EMBED[bd["index"]] == bd["embed"], "COUPLED_INDEX_MISMATCH"
        return "ATOMIC_ROLLBACK", rb_ptr, alerts
    n_req = min_canary_n(0.95, 0.01)
    if n2 < n_req:
        return f"HOLD({n2}/{n_req})", active_ptr, []
    return "PROMOTE", active_ptr, []

base_slo = {"n": 6000, "empty_rate": 0.048, "downvote_rate": 0.021}
windows = [
    ("5%灰度冷启动", {"n": 180, "ttft_p99_ms": 625, "tpot_p99_ms": 18.2, "tool_err_rate": 0.006, "empty_rate": 0.050, "nli_low_rate": 0.028, "downvote_rate": 0.022}),
    ("凌晨日志突变", {"n": 320, "ttft_p99_ms": 640, "tpot_p99_ms": 18.6, "tool_err_rate": 0.034, "empty_rate": 0.241, "nli_low_rate": 0.062, "downvote_rate": 0.041}),
    ("修复后25%灰度", {"n": 1200, "ttft_p99_ms": 638, "tpot_p99_ms": 18.4, "tool_err_rate": 0.005, "empty_rate": 0.049, "nli_low_rate": 0.026, "downvote_rate": 0.020}),
]
for name, w in windows:
    act, ptr, al = canary_guard_and_rollback("bundle_v20", base_slo, w)
    print(f"{name} | action={act:15s} | active={ptr} | alerts={al}")
