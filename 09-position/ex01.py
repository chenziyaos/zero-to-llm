# ZTL-09｜模型｜让模型知道顺序 · PyTorch 复现 Qwen3.8 风格 25% PartialRoPE (64/256 维)、相对位移不变性与左填充增量缓存对齐
import torch

def precompute_partial_rope(d_rot: int = 64, max_seq: int = 1024, base: float = 10_000_000.0):
    # 强制在 float32 下计算 32 对逆频率与外积相位，杜绝 BF16 大整数尾数跳变
    inv_freq = 1.0 / (base ** (torch.arange(0, d_rot, 2, dtype=torch.float32) / d_rot))
    freqs = torch.outer(torch.arange(max_seq, dtype=torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)  # 半段切分排布: (max_seq, 64)
    return emb.cos(), emb.sin()

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def apply_partial_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                       pos_ids: torch.Tensor, d_rot: int = 64):
    # 仅切前 d_rot=64 维做正交旋转，后 192 维纯内容语义原样透传 (省 75% 旋转开销)
    c = cos[pos_ids].unsqueeze(1).to(q.dtype)
    s = sin[pos_ids].unsqueeze(1).to(q.dtype)
    q_rot = (q[..., :d_rot] * c) + (rotate_half(q[..., :d_rot]) * s)
    k_rot = (k[..., :d_rot] * c) + (rotate_half(k[..., :d_rot]) * s)
    return torch.cat([q_rot, q[..., d_rot:]], dim=-1), torch.cat([k_rot, k[..., d_rot:]], dim=-1)

if __name__ == "__main__":
    cos_t, sin_t = precompute_partial_rope(d_rot=64, max_seq=64, base=10_000_000.0)
    q = torch.full((1, 1, 1, 256), 0.5)
    k = torch.linspace(-0.5, 0.5, 256).view(1, 1, 1, 256)
    q5, _ = apply_partial_rope(q, k, cos_t, sin_t, torch.tensor([[5]]))
    _, k2 = apply_partial_rope(q, k, cos_t, sin_t, torch.tensor([[2]]))
    q10, _ = apply_partial_rope(q, k, cos_t, sin_t, torch.tensor([[10]]))
    _, k7 = apply_partial_rope(q, k, cos_t, sin_t, torch.tensor([[7]]))
    # 核验 (5,2) 与 (10,7) 全 256 维点积均为 2.00921，且后 192 维纯内容零偏移
    assert torch.allclose((q5 * k2).sum(), (q10 * k7).sum(), atol=1e-6)
    assert torch.equal(q5[..., 64:], q10[..., 64:])

    # 左填充批解码对齐：样本 0 含 2 个左 PAD + 3 个有效词，样本 1 含 5 个有效词
    attn_mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    prefill_pos = (attn_mask.cumsum(dim=-1) - 1).clamp(min=0)      # [[0,0,0,1,2], [1..4]]
    decode_pos = attn_mask.sum(dim=-1, keepdim=True)               # 首步新词续接 [[3], [5]]
    print("Dot(5,2)  :", round((q5 * k2).sum().item(), 6))
    print("Dot(10,7) :", round((q10 * k7).sum().item(), 6))
    print("prefill pos:", prefill_pos.tolist())
    print("decode pos :", decode_pos.tolist())
