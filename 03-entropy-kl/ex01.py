# ZTL-03｜数学｜熵、交叉熵、困惑度与 KL · 在 BF16 半精度下复现显式 Softmax+log 下溢崩溃与 F.cross_entropy 平移 LogSumExp 安全输出
import torch
import torch.nn.functional as F

# 构造 BF16 极端打分：正确标签 y=2 落在极低分 -100.0，模拟冷僻转折词或噪声初期
z_naive = torch.tensor([[100.0, 0.0, -100.0]], dtype=torch.bfloat16, requires_grad=True)
z_fused = z_naive.detach().clone().requires_grad_(True)
target = torch.tensor([2])

# 1. 危险反例：先算显式 softmax 概率再取 -log，exp(-200) 在 BF16 下溢成 0.0
probs = F.softmax(z_naive, dim=-1)
loss_naive = -torch.log(probs[0, target[0]])
loss_naive.backward()
print(f"朴素拆算 -> q_y={probs[0, 2].item()}, loss={loss_naive.item()}, grad={z_naive.grad.tolist()}")

# 2. 安全反例：用 F.cross_entropy 的平移 LogSumExp 原位融合
z_fused = z_naive.detach().clone().requires_grad_(True)
loss_fused = F.cross_entropy(z_fused, target)
loss_fused.backward()
print(f"融合 LogSumExp -> loss={loss_fused.item()}, grad={z_fused.grad.tolist()}")
print(f"结论：指数下溢导致 NaN 传播失败，LogSumExp 融合确保数值安全与有界梯度")
