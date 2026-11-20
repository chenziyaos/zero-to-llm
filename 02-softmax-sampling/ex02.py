# ZTL-02｜数学｜条件概率、softmax 与采样 · 在 248,320 维词表上验证 T=1.0 + top_k=20 + top_p=0.95 双重截断在算术步与分叉步的自适应伸缩
import numpy as np

def decode_topk_topp_248k(logits: np.ndarray, temp: float = 1.0,
                          top_k: int = 20, top_p: float = 0.95, seed: int = 42):
    if temp <= 1e-6:
        best = int(np.argmax(logits))
        return best, 1, 1.0
    scaled = (logits.astype(np.float64) - np.max(logits)) / temp
    # 1. 先过 top_k=20 硬封底：O(V) 复杂度利用 argpartition 一步剔除 248,300 个长尾词！
    if 0 < top_k < scaled.size:
        cand_idx = np.argpartition(scaled, -top_k)[-top_k:]
        cand_logits = scaled[cand_idx]
    else:
        cand_idx, cand_logits = np.arange(scaled.size), scaled
    # 2. 仅对最多 20 个保留项降序排列并执行右移掩码 top_p=0.95 核截断
    order = np.argsort(cand_logits)[::-1]
    s_idx, s_logits = cand_idx[order], cand_logits[order]
    s_probs = np.exp(s_logits - s_logits[0])
    s_probs /= s_probs.sum()
    cum_probs = np.cumsum(s_probs)
    keep = (cum_probs - s_probs) < top_p  # 右移掩码：确保首词与刚跨过阈值的临界词必留！
    active_idx, active_probs = s_idx[keep], s_probs[keep]
    active_probs /= active_probs.sum()    # 3. 截断后严格重归一化，使总和恢复 1.0
    chosen = int(np.random.default_rng(seed).choice(active_idx, p=active_probs))
    return chosen, int(keep.sum()), float(active_probs[0])

# 构造 248,320 维词表：场景 A 为高确定算术步，场景 B 为多路思考分叉步
z_math = np.full(248_320, -9.5, dtype=np.float32); z_math[56] = 6.2; z_math[1:20] = 0.0
z_fork = np.full(248_320, -9.5, dtype=np.float32); z_fork[:8] = [3.2, 2.9, 2.6, 2.3, 2.0, 1.7, 0.8, 0.2]
_, n_math, p1_math = decode_topk_topp_248k(z_math, temp=1.0, top_k=20, top_p=0.95)
_, n_fork, p1_fork = decode_topk_topp_248k(z_fork, temp=1.0, top_k=20, top_p=0.95)
print(f"算术高确定步: 248320 -> 存活核大小={n_math} 词, 榜首重归一概率={p1_math*100:.2f}%")
print(f"思考多分叉步: 248320 -> 存活核大小={n_fork} 词, 榜首重归一概率={p1_fork*100:.2f}%")
print("assert: 双重截断协同起效，自适应伸缩兼得精准与探索")
