# ZTL-19｜系统｜判断慢在哪，以及请求怎样排队 · 分块混批 + 异步重叠离散事件模拟器与单卡 H100 案例模型线上排障三阶段对账脚本
def sim_step_scheduler(chunk_budget=None, async_overlap=False):
    decodes = [16] * 12                       # 12 路在线流式 Decode，各剩 16 词待吐
    prefill_rem, arrived = 0, False
    t_ms, last_emit, max_itl = 0.0, [0.0] * 12, 0.0
    cpu_bubble = 0.0 if async_overlap else 2.4
    while any(d > 0 for d in decodes) or prefill_rem > 0 or not arrived:
        if not arrived and t_ms >= 36.0:
            prefill_rem, arrived = 32768, True   # t=36ms 突入一条 32K 仓库审阅请求
        act_d = [i for i, d in enumerate(decodes) if d > 0]
        if chunk_budget is None and prefill_rem > 0:
            p_tok, d_run = prefill_rem, []       # 旧式 Prefill-First: 整块独占挂起全部解码
        else:
            d_run = act_d
            raw_b = max(0, (chunk_budget or 0) - len(d_run))
            p_tok = min(prefill_rem, (raw_b // 64) * 64 or raw_b)  # 对齐 64 词 GDN 块边界
        if p_tok > 0 and not d_run:
            gpu_ms = 0.0146 * p_tok
        elif p_tok > 0 and d_run:
            gpu_ms = max(16.2, 0.0098 * p_tok + 1.1)
        else:
            gpu_ms = 16.2 + 0.14 * len(d_run)
        t_ms += gpu_ms + cpu_bubble
        prefill_rem -= p_tok
        for i in d_run:
            if i < 12:
                max_itl = max(max_itl, t_ms - last_emit[i])
            last_emit[i] = t_ms
            decodes[i] -= 1
        if arrived and prefill_rem == 0 and len(decodes) == 12:
            decodes.append(6)
            last_emit.append(t_ms)
    return round(t_ms, 1), round(max_itl, 1)

for tag, cb, ov in [("故障基线: Prefill-First + 同步排批", None, False),
                    ("仅开分块: Chunked 2048 + 同步排批",  2048, False),
                    ("分块+重叠: Chunked 2048 + 异步重叠", 2048, True)]:
    tot, itl = sim_step_scheduler(cb, ov)
    print(f"{tag:32s} | 批次完工:{tot:6.1f}ms | 老用户 max_ITL:{itl:5.1f}ms")

# 三阶段验收账单: 下列 P99 / 抢占 / Goodput 为假设读数（示意，非实测）
GPU_USD_H = 3.00
STAGES = [
    ("故障态: 整块Prefill+预划死池+BF16",        520.0, 480.8, 38,  440),
    ("阶段一: 仅开2048分块+异步重叠(仍满池抢占)",  88.4,  28.5, 16, 1520),
    ("投产态: 分块重叠+统一双向池+FP8+对齐检查点", 23.8,  23.6,  0, 2260),
]
for name, tpot, itl, pre, good_tps in STAGES:
    usd_m = GPU_USD_H / (good_tps * 3600 / 1e6)
    print(f"{name:38s} | 抢占:{pre:2d}/分 | P99_ITL:{itl:5.1f}ms | "
          f"P99_TPOT:{tpot:5.1f}ms | Goodput:{good_tps:4d} tok/s | ${usd_m:.2f}/M")
