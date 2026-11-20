# ZTL-13｜后训练｜奖励、策略梯度与 DPO 的数学 · PyTorch 完整可运行实现：GRPO 组内优势归一化、双侧裁剪、Schulman k3 散度与 Miles TIS 训推修正
import torch

def grpo_tis_loss(logp_new, logp_old, logp_rollout, logp_ref, mask, rewards, eps=0.2, beta=0.04, tis_cap=2.0):
    mu = rewards.mean(dim=1, keepdim=True)
    sigma = rewards.std(dim=1, unbiased=False, keepdim=True)
    adv = ((rewards - mu) / (sigma + 1e-6)).unsqueeze(-1)            # 组内归一化并广播至词元维
    rho = torch.exp(logp_new - logp_old)
    surr = torch.minimum(rho * adv, torch.clamp(rho, 1.0 - eps, 1.0 + eps) * adv)
    tis_w = torch.clamp(torch.exp(logp_old.detach() - logp_rollout), min=0.5, max=tis_cap)
    log_q = logp_ref - logp_new                                     # Schulman k3: exp(Δ)-Δ-1 >= 0
    kl_k3 = torch.exp(log_q) - log_q - 1.0
    tok_loss = (-tis_w * surr + beta * kl_k3) * mask
    seq_loss = tok_loss.sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
    drift = ((logp_old.detach() - logp_rollout).abs() * mask).sum() / mask.sum()
    return seq_loss.mean(), adv[:, :, 0], (kl_k3 * mask).sum() / mask.sum(), drift

rewards = torch.tensor([[1.0, 1.0, 0.0, 0.0]])                      # 同题 G=4 条轨迹: 2 对 2 错
mask = torch.tensor([[[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]])
logp_roll = torch.tensor([[[-1.10, -0.85, -0.60], [-0.95, -0.80, 0.0], [-1.40, -1.05, -0.90], [-1.00, -1.25, 0.0]]])
logp_old = logp_roll + 0.02                                         # BF16 重算相对 FP8 采样的 0.02 nats 漂移
logp_ref = logp_old - 0.05
delta = torch.tensor([[[0.25, 0.08, 0.05], [0.06, 0.04, 0.0], [-0.14, -0.10, -0.08], [-0.28, -0.10, 0.0]]])
logp_new = (logp_old + delta).detach().requires_grad_(True)
loss, adv, mean_kl, drift = grpo_tis_loss(logp_new, logp_old, logp_roll, logp_ref, mask, rewards)
loss.backward()
print(f"组内优势 A_i : {adv[0].tolist()} | loss={loss.item():.4f} | k3_kl={mean_kl.item():.4f} | drift={drift.item():.4f}")
print(f"正优势首词超 +0.2 裁剪后梯度: {logp_new.grad[0, 0].round(decimals=4).tolist()}")
print(f"负优势首词超 -0.2 裁剪后梯度: {logp_new.grad[0, 3].round(decimals=4).tolist()}")
