# ZTL-29｜工程｜Agent 的循环、记忆、权限与成本 · 带步数预算硬顶、规范化动作指纹拦截与半途结构化交接的受控 ReAct 状态机
import hashlib
import json

PRICE = {"miss": 2.00, "hit": 0.20, "out": 8.00}  # 美元 / 百万词元

def action_fingerprint(tool: str, args: dict) -> str:
    clean = {k: v for k, v in args.items() if k not in ("req_id", "trace_ts")}
    canonical = json.dumps(clean, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{tool}:{canonical}".encode()).hexdigest()[:12]

def run_controlled_react(steps, s0_tok=3600, t_max=10, usd_cap=0.35):
    seen_fp, consec_err, verified_facts = {}, 0, []
    cost_usd, ctx_tok, prev_delta = 0.0, s0_tok, 0
    for t in range(1, t_max + 1):
        tool, args, out_tok, obs_tok, ok, summary = steps[min(t - 1, len(steps) - 1)]
        hit_tok = ctx_tok if t > 1 else 0
        miss_tok = s0_tok if t == 1 else prev_delta
        cost_usd += (miss_tok * PRICE["miss"] + hit_tok * PRICE["hit"] + out_tok * PRICE["out"]) / 1e6
        if tool == "finish_diagnosis":
            return {"status": "RESOLVED", "turns": t, "usd": round(cost_usd, 4), "diagnosis": summary}
        # 第一级熔断：抵达 T_max=10 或累计成本触及 85% 警戒线，转入半途交接
        if t == t_max or cost_usd >= 0.85 * usd_cap:
            return {"status": "PARTIAL_HANDOFF", "turns": t, "usd": round(cost_usd, 4),
                    "facts": verified_facts[:3], "blocked_by": "budget_or_step_cap",
                    "human_cmd": "kubectl top pod -n prod && check upstream envoy"}
        # 第二级熔断：规范化指纹判重 + 连续报错计数
        fp = action_fingerprint(tool, args)
        seen_fp[fp] = seen_fp.get(fp, 0) + 1
        consec_err = 0 if ok else consec_err + 1
        if seen_fp[fp] >= 2 or consec_err >= 3:
            if seen_fp[fp] >= 3 or consec_err >= 3:
                # 第三级兜底：清空工具表，输出半途结构化交接单而非抛 500 白屏
                return {"status": "PARTIAL_HANDOFF", "turns": t, "usd": round(cost_usd, 4),
                        "facts": verified_facts[:3], "blocked_by": f"{tool}: {summary}",
                        "human_cmd": "kubectl auth can-i get secrets -n prod --as=sre-oncall"}
            obs_tok = 140  # 第 2 次同参撞墙：拦截调用，注入 140 词纠偏提示强制换路
        elif ok:
            verified_facts.append(summary)
        prev_delta = out_tok + obs_tok
        ctx_tok += prev_delta

trace_ok = [
    ("query_apm", {"svc": "pay-gw", "win": "5m"}, 220, 260, True, "pay-gw P99=4.8s, db-proxy 504"),
    ("kubectl_logs_ptr", {"pod": "db-proxy-2"}, 190, 210, True, "saved /workspace/obs2.log, 37 PoolExhausted"),
    ("read_span", {"path": "/workspace/obs2.log", "s": 410, "e": 445}, 180, 320, True, "maxIdle=200 leaked on retry"),
    ("finish_diagnosis", {}, 260, 0, True, "db-proxy retry leak exhausted 200 conn pool"),
]
trace_403 = [
    ("query_apm", {"svc": "auth-svc", "win": "5m"}, 200, 240, True, "auth-svc 403 spike after key-rot-v4"),
    ("read_k8s_secret", {"ns": "prod", "name": "jwt-key", "req_id": "r1"}, 180, 120, False, "403 Forbidden RBAC"),
    ("read_k8s_secret", {"name": "jwt-key", "ns": "prod", "req_id": "r2"}, 180, 120, False, "403 Forbidden RBAC"),
    ("read_k8s_secret", {"ns": "prod", "name": "jwt-key", "req_id": "r3"}, 180, 120, False, "403 Forbidden RBAC"),
]
for name, trace in (("正常定位", trace_ok), ("熔断交接", trace_403)):
    r = run_controlled_react(trace)
    print(f"{name}: status={r['status']} turns={r['turns']} usd={r['usd']}")
    for k in ("diagnosis", "facts", "blocked_by", "human_cmd"):
        if k in r:
            print(f"  {k}: {r[k]}")
