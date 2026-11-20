# ZTL-12｜训练｜多卡的账和中断后续上 · PyTorch DCP 三阶段原子提交范式与第 10 步断电续跑至第 20 步逐比特零误差验证
import math, os, pickle, shutil, tempfile
import numpy as np

# 生产多机范式（在 torchrun 分布式环境中传入 torch.distributed.checkpoint 与 dist 调用）：
def save_dcp_atomic(dcp_mod, dist_mod, state_5tuple, ckpt_dir):
    tmp_dir = ckpt_dir + ".step_tmp"
    dcp_mod.save(state_5tuple, checkpoint_id=tmp_dir)  # 阶段1：各 Rank 并行直写 .step_tmp
    dist_mod.barrier()                                 # 阶段2：全员屏障确认无任何分片掉队
    if dist_mod.get_rank() == 0:
        os.replace(tmp_dir, ckpt_dir)                  # 阶段3：Rank 0 原子改名瞬间转正
    dist_mod.barrier()

def adamw_step(w, m, v, t, x, y, lr, rng):
    keep = (rng.random(w.shape[0]) >= 0.2) / 0.8       # 强依赖各 Rank 独立随机种子态
    xm = x * keep
    logits = np.einsum("bi,ic->bc", xm, w)
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    loss = float(-np.log(probs[np.arange(len(y)), y]).mean())
    probs[np.arange(len(y)), y] -= 1.0
    g = np.einsum("bi,bc->ic", xm, probs / len(y))
    m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * (g * g)
    mh, vh = m / (1.0 - pow(0.9, t)), v / (1.0 - pow(0.999, t))
    w = w * (1.0 - lr * 0.01) - lr * mh / (np.sqrt(vh) + 1e-8)
    return w, m, v, loss

data_rng = np.random.default_rng(7)
stream = [(data_rng.standard_normal((16, 32)), data_rng.integers(0, 8, 16)) for _ in range(20)]
cos_lr = lambda s: 5e-2 * 0.5 * (1.0 + math.cos(math.pi * s / 20))
root, rng_ref = tempfile.mkdtemp(), np.random.default_rng(42)
ckpt10 = os.path.join(root, "ckpt_step_010")
w0 = rng_ref.standard_normal((32, 8)) * 0.2
w, m, v, ref_losses = w0.copy(), np.zeros_like(w0), np.zeros_like(w0), []

for s in range(20):                                    # 基准组：连续无中断跑满 1..20 步
    w, m, v, l = adamw_step(w, m, v, s + 1, *stream[s], cos_lr(s), rng_ref)
    ref_losses.append(l)
    if s + 1 == 10:                                    # 第 10 步执行三阶段 .step_tmp 原子转正
        tmp = ckpt10 + ".step_tmp"
        os.makedirs(tmp, exist_ok=True)
        with open(os.path.join(tmp, "shard0.pkl"), "wb") as f:
            pickle.dump({"w": w, "m": m, "v": v, "step": 10, "cursor": 10,
                         "rng": rng_ref.bit_generator.state}, f)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, ckpt10)
        os.makedirs(os.path.join(root, "ckpt_step_011.step_tmp"), exist_ok=True)  # 模拟11步写盘断电坏档

valid = sorted(d for d in os.listdir(root) if not d.endswith(".step_tmp"))[-1]
st = pickle.load(open(os.path.join(root, valid, "shard0.pkl"), "rb"))

wb, mb, vb, rbad, bad_losses = st["w"].copy(), np.zeros_like(w0), np.zeros_like(w0), np.random.default_rng(99), []
for idx, s in enumerate([0, 1] + list(range(12, 20))): # 反例：漏存游标重读旧批 + 漏存双矩符号跳刺
    wb, mb, vb, l = adamw_step(wb, mb, vb, idx + 1, *stream[s], cos_lr(0), rbad)
    bad_losses.append(l)

wo, mo, vo, rok, ok_losses = st["w"].copy(), st["m"].copy(), st["v"].copy(), np.random.default_rng(0), []
rok.bit_generator.state = st["rng"]
for s in range(st["cursor"], 20):                      # 正例：完整五元组恢复续跑第 11..20 步
    wo, mo, vo, l = adamw_step(wo, mo, vo, s + 1, *stream[s], cos_lr(s), rok)
    ok_losses.append(l)

shutil.rmtree(root)
max_diff = max(abs(a - b) for a, b in zip(ref_losses[10:], ok_losses))
print("连续真值 step 11..14:", [round(x, 4) for x in ref_losses[10:14]], "| 末步20:", round(ref_losses[-1], 4))
print("仅存权重 step 11..14:", [round(x, 4) for x in bad_losses[:4]],   "| 末步20:", round(bad_losses[-1], 4))
print("五元组续 step 11..14:", [round(x, 4) for x in ok_losses[:4]],    "| 末步20:", round(ok_losses[-1], 4))
print("max_diff ==", max_diff, "| 第20步最大权重绝对差:", float(np.max(np.abs(w - wo))))
