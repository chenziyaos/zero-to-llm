# ZTL-08｜模型｜把注意力写成矩阵 · 自包含 PyTorch Gated-GQA 模块（含独立头宽 d_k=256、QK-Norm、输出门控与 Prefill/Decode 误差 < 1e-5 单测）
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedGQAAttention(nn.Module):
    def __init__(self, d_model: int, h_q: int, h_kv: int, d_k: int):
        super().__init__()
        assert h_q % h_kv == 0, "查询头数 h_q 必须被键值头数 h_kv 整除"
        self.h_q, self.h_kv, self.d_k = h_q, h_kv, d_k
        self.groups = h_q // h_kv
        self.wq_gate = nn.Linear(d_model, 2 * h_q * d_k, bias=False)  # 合并投影 Q 与输出门控
        self.wk = nn.Linear(d_model, h_kv * d_k, bias=False)
        self.wv = nn.Linear(d_model, h_kv * d_k, bias=False)
        self.q_norm = nn.RMSNorm(d_k, eps=1e-6)
        self.k_norm = nn.RMSNorm(d_k, eps=1e-6)
        self.wo = nn.Linear(h_q * d_k, d_model, bias=False)

    def forward(self, x: torch.Tensor, kv_cache=None):
        b, l_q, _ = x.shape
        q, gate = self.wq_gate(x).chunk(2, dim=-1)
        q = self.q_norm(q.view(b, l_q, self.h_q, self.d_k)).transpose(1, 2)
        k = self.k_norm(self.wk(x).view(b, l_q, self.h_kv, self.d_k)).transpose(1, 2)
        v = self.wv(x).view(b, l_q, self.h_kv, self.d_k).transpose(1, 2)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        l_k = k.size(2)
        # 教学演示显式广播；生产内核开启 F.scaled_dot_product_attention(..., enable_gqa=True) 零拷贝读取
        k_rep = k.repeat_interleave(self.groups, dim=1)
        v_rep = v.repeat_interleave(self.groups, dim=1)
        scores = torch.matmul(q, k_rep.transpose(-2, -1)) / math.sqrt(self.d_k)
        if l_q > 1:  # 右下对齐因果掩码: j <= i + (L_k - L_q)
            i = torch.arange(l_q, device=x.device).unsqueeze(1)
            j = torch.arange(l_k, device=x.device).unsqueeze(0)
            scores = scores.masked_fill(j > i + (l_k - l_q), float("-inf"))
        ctx = torch.matmul(F.softmax(scores, dim=-1), v_rep)
        ctx = ctx.transpose(1, 2).contiguous().view(b, l_q, -1)
        return self.wo(ctx * torch.sigmoid(gate)), (k, v)

torch.manual_seed(42)
attn = GatedGQAAttention(d_model=640, h_q=3, h_kv=1, d_k=256).eval()
x = torch.randn(2, 6, 640)
with torch.no_grad():
    y_prefill, _ = attn(x)                             # 6 词整句并行 Prefill
    _, cache = attn(x[:, :5])                          # 前 5 词写入单头 KV Cache
    y_decode, cache = attn(x[:, 5:6], kv_cache=cache)  # 第 6 词单步增量 Decode (l_q=1 免掩码)
assert cache[0].shape == (2, 1, 6, 256)
err = (y_prefill[:, -1:] - y_decode).abs().max().item()
print(f"测试 Gated-GQA：3 查询头分享 1 键值头，d_k=256（非 64/128 标准宽）")
print(f"整句预填充末词 shape: {y_prefill[:, -1:].shape} | 单步解码 shape: {y_decode.shape}")
print(f"KV 缓存头维 (B, h_kv, S, d_k): {cache[0].shape} | Prefill-vs-Decode 误差: {err:.2e}")
assert err < 1e-5  # 实测误差 ~2.38e-07，断言稳过
