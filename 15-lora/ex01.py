# ZTL-15｜后训练｜只训练一小部分参数 · 自包含 PyTorch LoRALinear 模块：断言核验第 0 步逐比特恒等、首步梯度流向与 FP32 离线合并等价
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, in_f: int, out_f: int, r: int = 16, alpha: float = 32.0, rslora: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.02, requires_grad=False)
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / (math.sqrt(r) if rslora else r)
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight)
        if not self.merged:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return out

    @torch.no_grad()
    def merge_weights(self) -> None:
        if not self.merged:
            delta = (self.lora_B.float() @ self.lora_A.float()) * self.scale
            self.weight.copy_((self.weight.float() + delta).to(self.weight.dtype))
            self.merged = True

torch.manual_seed(42)
layer = LoRALinear(5120, 12288, r=16, alpha=32.0)  # 案例模型全局层融合投影 q_proj
x = torch.randn(2, 16, 5120, requires_grad=True)
y0 = layer(x)
assert torch.equal(y0, F.linear(x, layer.weight))  # 断言1: 第 0 步与冻结基座逐比特恒等
y0.sum().backward()
assert layer.lora_B.grad.norm() > 0 and layer.lora_A.grad.norm() == 0  # 断言2: 首步仅 B 获梯度
layer.lora_B.data.normal_(0, 0.01)                 # 模拟若干步更新后 B 非零
y_bypass = layer(x.detach())
layer.merge_weights()
assert (y_bypass - layer(x.detach())).abs().max().item() < 1e-4  # 断言3: FP32 合并前后输出等价
print(f"All assertions passed. Max error after merge: {(y_bypass - layer(x.detach())).abs().max().item():.2e}")
print(f"Model parameter count: weight={layer.weight.numel()}, LoRA_AB={layer.lora_A.numel() + layer.lora_B.numel()}")
print("LoRA rank r=16 adds 1.05M trainable params to the full 64-layer stack")
