# ZTL-23｜系统｜把模型部署成服务 · 仿真 24 QPS 突发洪峰下无界排队傻跑 vs 队列上限快返 429 配断连即回收的 Goodput 差距
import math, random

def sim_serving_guard(qps: float, max_q: int, abort_on_disc: bool, seed: int = 42, horizon: float = 300.0):
    rng = random.Random(seed)
    t, busy_until, q = 0.0, 0.0, []
    ok, slo_viol, rej_429, saved_ms = 0, 0, 0, 0.0
    while t < horizon:
        t += rng.expovariate(qps)
        q = [ft for ft in q if ft > t]
        if len(q) >= max_q:          # 队列触顶：2ms 内快返 HTTP 429 阻断积压传染
            rej_429 += 1
            continue
        svc = rng.lognormvariate(math.log(0.046), 0.42)
        client_cancelled = (rng.random() < 0.08)   # 8% 请求中途由用户主动断连
        eff_svc = svc * 0.25 if (client_cancelled and abort_on_disc) else svc
        if client_cancelled and abort_on_disc:
            saved_ms += (svc - eff_svc) * 1e3
        start = max(t, busy_until)
        ttft = (start - t) + eff_svc
        busy_until = start + eff_svc
        q.append(busy_until)
        if not client_cancelled:
            ok += int(ttft <= 0.50)  # 严核 P99 TTFT <= 500ms 红线
            slo_viol += int(ttft > 0.50)
    return round(ok / horizon, 2), round(slo_viol / horizon, 2), round(rej_429 / horizon, 2), round(saved_ms / 1e3, 1)

print("=== Scenario 1: Unbounded Queue ===")
r1 = sim_serving_guard(24.0, max_q=9999, abort_on_disc=False)
print("无界队列+断连傻跑:", r1)
print("=== Scenario 2: MaxQ=6 + Early Abort ===")
r2 = sim_serving_guard(24.0, max_q=6, abort_on_disc=True)
print("MaxQ=6 +断连即回收:", r2)
