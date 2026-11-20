# ZTL-03｜数学｜熵、交叉熵、困惑度与 KL · 验证双峰分布前反向 KL 性格反转、PyTorch F.kl_div 参数契约与 Schulman k3 估计器低方差特性
import torch
import torch.nn.functional as F

p = torch.tensor([0.500, 0.005, 0.495])
q_cov, q_seek = torch.tensor([0.35, 0.30, 0.35]), torch.tensor([0.98, 0.01, 0.01])

# 1. PyTorch 契约警告：F.kl_div(input=log_分母, target=分子, reduction='batchmean') 计算 D_KL(target || input)
fwd_cov = F.kl_div(q_cov.unsqueeze(0).log(), p.unsqueeze(0), reduction="batchmean").item()
fwd_seek = F.kl_div(q_seek.unsqueeze(0).log(), p.unsqueeze(0), reduction="batchmean").item()
rev_cov = F.kl_div(p.unsqueeze(0).log(), q_cov.unsqueeze(0), reduction="batchmean").item()
rev_seek = F.kl_div(p.unsqueeze(0).log(), q_seek.unsqueeze(0), reduction="batchmean").item()
print(f"铺平 q_cov  -> 前向 D(p||q)={fwd_cov:.4f} (胜), 反向 D(q||p)={rev_cov:.4f}")
print(f"锁峰 q_seek -> 前向 D(p||q)={fwd_seek:.4f},      反向 D(q||p)={rev_seek:.4f} (胜)")

# 2. 50,000 次在线采样对比反向 KL D(pi_theta || pi_ref) 下朴素 k1 与 Schulman k3 的标准差
torch.manual_seed(42)
pi_theta, pi_ref = torch.tensor([0.55, 0.25, 0.20]), torch.tensor([0.50, 0.30, 0.20])
idx = torch.multinomial(pi_theta, num_samples=50_000, replacement=True)
r = pi_ref[idx] / pi_theta[idx]
k1, k3 = -torch.log(r), (r - 1.0) - torch.log(r)
true_kl = (pi_theta * (pi_theta / pi_ref).log()).sum().item()
print(f"真实反向 KL={true_kl:.5f} | k1 均值={k1.mean():.5f}, 标准差={k1.std():.4f}, 含负值={(k1 < 0).any().item()}")
print(f"真实反向 KL={true_kl:.5f} | k3 均值={k3.mean():.5f}, 标准差={k3.std():.4f}, 最小值={k3.min():.5f}")
