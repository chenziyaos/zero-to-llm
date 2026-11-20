# ZTL-14｜后训练｜只对该学的回答算损失 · PyTorch 验证微批 mean 千倍权重畸变 vs 全局 ∑m_t 归约零误差等价及吞吐核算
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(7)
proj = nn.Linear(16, 64, bias=False).double()
nn.init.zeros_(proj.weight)  # 零权重点处 softmax 严格均匀分布，消除跨版本随机种子差异
# 微批 1: 长背景 + "Yes<|im_end|>" 仅 M1=2 个监督词；微批 2: 长多轮排障轨迹 M2=2000 个监督词
h1, y1 = torch.full((32, 16), 0.5, dtype=torch.float64), torch.tensor([-100] * 30 + [11, 63])
h2, y2 = torch.full((2048, 16), -0.5, dtype=torch.float64), torch.tensor([-100] * 48 + [25] * 1999 + [63])

# 基准真值: 无累积单次合并全批反传
proj.zero_grad()
F.cross_entropy(proj(torch.cat([h1, h2])), torch.cat([y1, y2]), ignore_index=-100, reduction="mean").backward()
g_true = proj.weight.grad.clone()

# 缺陷写法: 2 个微批各自 reduction="mean" 除以 K=2 累积 (M1 单词权重被放大 1000 倍!)
proj.zero_grad()
(F.cross_entropy(proj(h1), y1, ignore_index=-100, reduction="mean") / 2.0).backward()
(F.cross_entropy(proj(h2), y2, ignore_index=-100, reduction="mean") / 2.0).backward()
g_bug = proj.weight.grad.clone()

# 正确写法: 微批内 reduction="sum"，统一除以全局有效监督词总数 N_valid = M1 + M2 = 2002
proj.zero_grad()
n_valid = int((y1 != -100).sum() + (y2 != -100).sum())
(F.cross_entropy(proj(h1), y1, ignore_index=-100, reduction="sum") / n_valid).backward()
(F.cross_entropy(proj(h2), y2, ignore_index=-100, reduction="sum") / n_valid).backward()
g_fix = proj.weight.grad.clone()

cos_bug = float(F.cosine_similarity(g_bug.flatten(), g_true.flatten(), dim=0))
print(f"N_valid={n_valid} | Bug 方向余弦={cos_bug:.4f}, 最大误差={(g_bug - g_true).abs().max():.4f}")
print(f"Fix 全局归约最大绝对误差={(g_fix - g_true).abs().max():.1e} (机器精度严格全等)")

# Qwen3.8-27B (P=27B, D_real=2.808e8) 在 8×H100 (989 TFLOPS, $3/卡时) 排障前后账单核算
P, D_REAL, PEAK_TF, NGPU = 27.0e9, 120000 * 2340, 989e12, 8
for tag, raw_tps, fill in [("排障前(8K补PAD)", 2748.0, 2340 / 8192), ("排障后(双隔离打包)", 2757.0, 0.986)]:
    eff_tps = raw_tps * fill
    mfu = 6.0 * P * eff_tps / PEAK_TF * 100
    wall_h = D_REAL / eff_tps / 3600 / NGPU
    print(f"{tag}: 有效吞吐={eff_tps:.0f} tok/s | 真MFU={mfu:.1f}% | 8卡耗时={wall_h:.2f}h | 成本=${wall_h*NGPU*3:.1f}")
