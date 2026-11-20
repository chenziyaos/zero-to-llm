# ZTL-10｜模型｜组装小模型并让一个批次下降 · 零外部依赖验算二维 SwiGLU 门控前向、硬件倍数对齐及参数账本
import math

# 1. 手算复核二维到三维 SwiGLU 门控前向 (x = [1.0, -1.0]) 与 128/256 硬件对齐公式
silu = lambda z: z / (1.0 + math.exp(-z))
z_gate, u_up = [1.0, -1.0, 0.0], [1.0, -1.0, 1.0]
h_mid = [round(silu(z) * u, 4) for z, u in zip(z_gate, u_up)]
align_dff = lambda d, m: m * (((8 * d) // 3 + m - 1) // m)
print(f"SwiGLU 中间态 g⊙u = {h_mid} | 对齐 d_ff: d=256->{align_dff(256, 128)}, d=4096->{align_dff(4096, 256)}")

# 2. 逐项清点案例模型 (64层: 48 GDN + 16 Gated GQA) 与 MiniQwen3.8 (4层: 3 GDN + 1 Gated GQA)
q_vocab = 2 * 248320 * 5120
q_swiglu = 64 * 3 * 5120 * 17408
q_gqa = 16 * (5120 * 12288 + 2 * 5120 * 1024 + 6144 * 5120 + 512)
q_gdn = 48 * (5120 * 16384 + 6144 * 5120 + 5120 * 96 + 96 + 10240 * 4 + 6144)
q_norms = 129 * 5120
q_backbone = q_swiglu + q_gqa + q_gdn + 128 * 5120
q_total = q_vocab + q_swiglu + q_gqa + q_gdn + q_norms

m_vocab, m_swiglu = 2 * 4096 * 256, 4 * 3 * 256 * 768
m_gdn = 3 * (256 * 384 + 384 * 4 + 256 * 272 + 32 + 256 * 256)
m_gqa = 1 * (256 * 512 + 2 * 256 * 64 + 64 + 256 * 256)
m_total = m_vocab + m_swiglu + m_gdn + m_gqa + 9 * 256
print(f"案例模型主干总参: {q_total:,} ({q_total/1e9:.2f}B) | SwiGLU 占64层主干: {q_swiglu/q_backbone*100:.2f}%")
print(f"MiniQwen3.8 解耦总参: {m_total:,} (共享词表总参: {m_total - 4096*256:,}，立省 {4096*256/m_total*100:.2f}%)")
