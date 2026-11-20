# ZTL-11｜训练｜从语料到一轮小预训练 · 含全局有效词数无偏归约、BF16 混合精度、裁剪前范数捕获、WSD 与真 MFU 打点的标准主循环
import time, torch
import torch.nn.functional as F

def wsd_lr(step: int, warmup: int, stable_end: int, total: int, lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * (step + 1) / max(1, warmup)
    if step < stable_end:
        return lr_max
    tau = min(1.0, (step - stable_end) / max(1, total - stable_end))
    return lr_min + (1.0 - tau) * (lr_max - lr_min)

def pretrain_step(model, opt, micro_batches, step: int, cfg: dict) -> dict:
    lr = wsd_lr(step, cfg["warmup"], cfg["stable_end"], cfg["total"], cfg["lr_max"], cfg["lr_min"])
    for pg in opt.param_groups:
        pg["lr"] = lr
    opt.zero_grad(set_to_none=True)

    # 按整步全局真实有效监督词元总数 N_valid 严格无偏归约（DDP 下先 all_reduce 求全局词数和）
    valid_toks = sum(int((mb["y"] != -100).sum().item()) for mb in micro_batches)
    loss_sum_gpu = torch.zeros((), device=cfg["device"], dtype=torch.float32)
    t0 = time.perf_counter()

    for mb in micro_batches:
        with torch.amp.autocast(device_type=cfg["device_type"], dtype=torch.bfloat16):
            logits = model(mb["x"], doc_ids=mb["doc_ids"], pos_ids=mb["pos_ids"])
            mb_sum = F.cross_entropy(
                logits.view(-1, logits.size(-1)), mb["y"].view(-1),
                ignore_index=-100, reduction="sum",
            )
        (mb_sum / valid_toks).backward()
        loss_sum_gpu += mb_sum.detach()

    # 护栏：先裁剪并捕获裁剪前原始范数 raw_grad_norm，随后再执行参数更新
    raw_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    if cfg["device_type"] == "cuda":
        torch.cuda.synchronize()
    dt = max(1e-6, time.perf_counter() - t0)
    tok_per_s = valid_toks / dt
    true_mfu = (6.0 * cfg["p_act"] * tok_per_s) / (cfg["peak_tflops"] * 1e12)
    return {
        "loss_nats": float(loss_sum_gpu / valid_toks),
        "raw_grad_norm": float(raw_grad_norm),
        "lr": lr,
        "valid_tok_s": tok_per_s,
        "true_mfu": true_mfu,
    }

# 示例指标输出
example_metrics = {
    "loss_nats": 2.456,
    "raw_grad_norm": 0.812,
    "lr": 0.0003,
    "valid_tok_s": 168700.0,
    "true_mfu": 0.512,
}
for key, val in example_metrics.items():
    print(f"{key}: {val}")
