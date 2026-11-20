# ZTL-30｜交付｜交一个可复查的项目 · 五元组可复现交付清单（Reproducibility Manifest）哈希校验与双模态容差审计器
import hashlib, json, math

REQUIRED_TUPLES = (
    "model_weights_lock", "prompt_mcp_lock", "rag_index_lock",
    "engine_pool_lock", "golden_eval_lock"
)

def canonical_digest(locks: dict) -> str:
    raw = json.dumps(locks, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

def verify_reproducibility(manifest: dict, run_report: dict) -> tuple[bool, str, list[str]]:
    errs = []
    locks = manifest.get("locks", {})
    for key in REQUIRED_TUPLES:
        if not locks.get(key):
            errs.append(f"MISSING:{key}")
    n = run_report["n_total"]
    if run_report["temperature"] == 0.0:
        if run_report["exact_match_cnt"] / n < 0.995:
            errs.append(f"GREEDY_DRIFT:{run_report['exact_match_cnt']}/{n}<99.5%")
    else:
        if run_report.get("seed") is None:
            errs.append("MISSING_SEED_FOR_SAMPLING")
        n10, n01 = run_report["win_repro"], run_report["loss_repro"]
        se_pair = math.sqrt(max(n10 + n01, 1)) / n
        diff = abs(n10 - n01) / n
        if diff > 1.96 * se_pair:
            errs.append(f"SAMPLING_DRIFT:{diff:.4f}>{1.96*se_pair:.4f}")
    return (len(errs) == 0), canonical_digest(locks), errs

rc1_manifest = {"locks": {
    "model_weights_lock": {"id": "Qwen/Qwen3.8-27B-FP8", "rev": "floating-main"},
    "prompt_mcp_lock": {"prompt_blob": "a81f02c", "mcp_schema": "v2.0"},
    "golden_eval_lock": {"file": "payment_oncall_600.jsonl", "sha256": "7c91e4b2"},
}}
rc2_manifest = {"locks": {
    "model_weights_lock": {"id": "Qwen/Qwen3.8-27B-FP8", "sha256": "e3b0c442", "thinking": "medium_T1.0_p0.95_k20"},
    "prompt_mcp_lock": {"prompt_blob": "4d92b1e", "mcp_schema_sha256": "91a0f8c3"},
    "rag_index_lock": {"chunk_spec": "parent800_child200", "embed_id": "bge-m3-v1.2", "alias": "kb_pay_v20"},
    "engine_pool_lock": {"image": "vllm-v1@sha256:5f2d8a", "gdn_slots": 64, "kv_kib_tok": 64, "batch_inv": True},
    "golden_eval_lock": {"file": "payment_oncall_600.jsonl", "sha256": "7c91e4b2", "n": 600},
}}

cases = [
    ("rc1_初审包(T=0贪心)", rc1_manifest, {"n_total": 600, "temperature": 0.0, "exact_match_cnt": 516}),
    ("rc2_终审包(T=0贪心)", rc2_manifest, {"n_total": 600, "temperature": 0.0, "exact_match_cnt": 599}),
    ("rc2_终审包(T=1思考)", rc2_manifest, {"n_total": 600, "temperature": 1.0, "seed": 42, "win_repro": 7, "loss_repro": 5}),
]
for tag, mf, rep in cases:
    ok, dg, err = verify_reproducibility(mf, rep)
    print(f"{tag} | pass={str(ok):5s} | digest={dg} | errors={err}")
