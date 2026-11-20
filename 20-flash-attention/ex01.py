# ZTL-20｜系统｜减少注意力的内存搬运 · 35 行自包含分块因果 FlashAttention：验证在线三统计量递推零误差（max_diff < 1e-6）与因果整块跳过
import math, numpy as np

np.random.seed(42)
N, d, Br, Bc = 256, 128, 32, 32
Q = np.random.randn(N, d).astype(np.float32)
K = np.random.randn(N, d).astype(np.float32)
V = np.random.randn(N, d).astype(np.float32)
scale = 1.0 / math.sqrt(d)

def standard_causal_attn(Q, K, V):
    S = np.einsum("id,jd->ij", Q, K) * scale
    S = np.where(np.triu(np.ones((N, N), dtype=bool), k=1), -np.inf, S)
    P = np.exp(S - np.max(S, axis=-1, keepdims=True))
    return np.einsum("ij,jd->id", P / np.sum(P, axis=-1, keepdims=True), V)

def flash_causal_attn(Q, K, V, Br=32, Bc=32):
    O, L = np.zeros_like(Q), np.zeros((N, 1), dtype=np.float32)
    skipped, total = 0, (N // Br) * (N // Bc)
    for r0 in range(0, N, Br):
        Qi = Q[r0:r0 + Br]
        m_i = np.full((Br, 1), -np.inf, dtype=np.float32)
        l_i, O_tilde = np.zeros((Br, 1), dtype=np.float32), np.zeros((Br, d), dtype=np.float32)
        for c0 in range(0, N, Bc):
            if c0 > r0 + Br - 1:  # 右上角纯未来分块整块跳过，零 HBM 读取零乘加
                skipped += 1; continue
            Kj, Vj = K[c0:c0 + Bc], V[c0:c0 + Bc]
            Sij = np.einsum("id,jd->ij", Qi, Kj) * scale
            if c0 + Bc - 1 > r0:  # 仅主对角线穿过的分块执行逐元素因果屏蔽
                Sij = np.where(np.arange(c0, c0 + Bc)[None, :] > np.arange(r0, r0 + Br)[:, None], -np.inf, Sij)
            m_new = np.maximum(m_i, np.max(Sij, axis=-1, keepdims=True))
            alpha, P_tilde = np.exp(m_i - m_new), np.exp(Sij - m_new)
            l_i = alpha * l_i + np.sum(P_tilde, axis=-1, keepdims=True)
            O_tilde = alpha * O_tilde + np.einsum("ij,jd->id", P_tilde, Vj)
            m_i = m_new
        O[r0:r0 + Br], L[r0:r0 + Br] = O_tilde / l_i, m_i + np.log(l_i)
    return O, L, skipped, total

O_ref = standard_causal_attn(Q, K, V)
O_fa, L_fa, sk, tot = flash_causal_attn(Q, K, V, Br, Bc)
print(f"三统计量递推与因果块跳过验证")
print(f"max_diff={np.max(np.abs(O_ref - O_fa)):.2e}")
print(f"因果块跳过率：{sk}/{tot} ({sk/tot:.1%})")
