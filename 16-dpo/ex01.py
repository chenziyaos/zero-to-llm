# ZTL-16｜后训练｜用偏好对核对 DPO · PyTorch 2B 拼接分块掩码求和、DPO 四数值手算断言与 Qwen3.8-27B 显存核算
import torch
import torch.nn.functional as F

def chunked_concatenated_logps(logits: torch.Tensor, labels: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    shift_logits, shift_labels = logits[:, :-1, :], labels[:, 1:]  # 右移一格因果对齐
    seq_sums = []
    for s in range(0, shift_labels.size(1), chunk):                # 沿序列分块防 248,320 词表爆显存
        lg = shift_logits[:, s:s + chunk, :].float()
        lb = shift_labels[:, s:s + chunk]
        mask = (lb != -100)                                        # 仅有效回答词与首颗 EOS 为 True
        tok_lp = F.log_softmax(lg, dim=-1).gather(-1, lb.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        seq_sums.append((tok_lp * mask).sum(dim=-1))               # 铁律：沿序列维严格求总和!
    return torch.stack(seq_sums, dim=-1).sum(dim=-1)

def compute_dpo_loss(pi_logps: torch.Tensor, ref_logps: torch.Tensor, beta: float = 0.20):
    pi_w, pi_l = pi_logps.chunk(2, dim=0)
    ref_w, ref_l = ref_logps.chunk(2, dim=0)
    delta_w, delta_l = pi_w - ref_w, pi_l - ref_l
    margins = beta * (delta_w - delta_l)
    return -F.logsigmoid(margins).mean(), margins, beta * delta_w, beta * delta_l, (margins > 0).float().mean()

def _logit_vec(tid: int, lp: float, v: int = 4) -> torch.Tensor:
    p = torch.exp(torch.tensor(lp, dtype=torch.float64))
    pr = torch.full((v,), (1.0 - p) / (v - 1), dtype=torch.float64); pr[tid] = p
    return torch.log(pr).float()

labels = torch.tensor([[-100, -100, 1, 2, 3, -100, -100], [-100, -100, 1, 2, 1, 3, -100]])
logits = torch.zeros(2, 7, 4)
for pos, tid, lp in [(1, 1, -1.20), (2, 2, -1.60), (3, 3, -1.40)]: logits[0, pos] = _logit_vec(tid, lp)
for pos, tid, lp in [(1, 1, -1.50), (2, 2, -1.80), (3, 1, -1.70), (4, 3, -1.70)]: logits[1, pos] = _logit_vec(tid, lp)
pi_lp = chunked_concatenated_logps(logits, labels, chunk=2)
loss, m, rw, rl, acc = compute_dpo_loss(pi_lp, torch.tensor([-5.00, -6.00]), beta=0.20)
assert torch.allclose(pi_lp, torch.tensor([-4.20, -6.70]), atol=1e-5) and abs(loss.item() - 0.55436) < 1e-5
v_full = 2 * 4096 * 248320 * 10 / 1e9  # Qwen3.8-27B (2B=2, S=4096, V=248320) 未分块词表瞬态缓冲 20.34 GB
print(f"核对通过: pi={pi_lp.tolist()} | margin={m.item():.4f} | loss={loss.item():.5f} | acc={acc.item():.1f}")
print(f"两条回答的总对数概率对比: 正例={pi_lp[0].item():.2f} | 负例={pi_lp[1].item():.2f}")
print(f"H100显存(GB): 双基座={108+12.7+v_full:.1f} | 关LoRA未分块={54+12.7+v_full:.1f} | 256分块={54+12.7+v_full/16:.1f}")
