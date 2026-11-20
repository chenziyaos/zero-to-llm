# ZTL-12｜训练｜多卡的账和中断后续上 · Qwen3.8-27B（Ψ=27B）在 8 卡与 16 卡 H100 下的 ZeRO 阶梯及 HSDP 跨机通信核算
PSI_B = 27.0       # Qwen3.8-27B 稠密参数 27.0B (16Ψ = 432.0 GB)
BW_NV = 360.0      # H100 机内 NVLink4 单向实测有效 busbw (GB/s)
BW_IB = 42.0       # 跨机 400G RDMA 网卡单向实测有效 busbw (GB/s)

def audit_zero(n_gpus, nodes=1, hsdp_intra=None):
    ddp = 16.0 * PSI_B
    z1  = (4.0 + 12.0 / n_gpus) * PSI_B
    z2  = (2.0 + 14.0 / n_gpus) * PSI_B
    z3  = (16.0 / n_gpus) * PSI_B
    ring = (n_gpus - 1) / n_gpus
    comm_z12 = 4.0 * ring * PSI_B   # 2Ψ 元素 × 2B/元素 × (N-1)/N
    comm_z3  = 6.0 * ring * PSI_B   # 3Ψ 元素 × 2B/元素 × (N-1)/N
    bw = BW_NV if nodes == 1 else BW_IB
    print(f"N={n_gpus:2d} ({nodes}机) | DDP:{ddp:5.1f}G -> Z1:{z1:6.2f}G -> Z2:{z2:6.2f}G -> Z3:{z3:4.1f}G "
          f"| Z1/2通信:{comm_z12:5.1f}G ({comm_z12/bw:4.2f}s) | 扁平Z3:{comm_z3:5.1f}G ({comm_z3/bw:4.2f}s)")
    if hsdp_intra:
        inter = n_gpus // hsdp_intra
        m_hsdp = (16.0 / hsdp_intra) * PSI_B
        ag_rs_nv = 6.0 * ((hsdp_intra - 1) / hsdp_intra) * PSI_B
        ar_ib = 4.0 * ((inter - 1) / inter) * (PSI_B / hsdp_intra)
        t_raw = ag_rs_nv / BW_NV + ar_ib / BW_IB
        print(f"HSDP ({hsdp_intra}机内切×{inter}跨机同) | 常驻静态:{m_hsdp:4.1f}G | 机内NV:{ag_rs_nv:5.1f}G "
              f"+ 跨机IB仅:{ar_ib:4.2f}G | 裸通信压至:{t_raw:4.2f}s")

audit_zero(8, nodes=1)
audit_zero(16, nodes=2, hsdp_intra=8)
# 精确输出（单张 H100 80 GiB 物理红线为 85.90 GB）：
# N= 8 (1机) | DDP:432.0G -> Z1:148.50G -> Z2:101.25G -> Z3:54.0G | Z1/2通信: 94.5G (0.26s) | 扁平Z3:141.8G (0.39s)
# N=16 (2机) | DDP:432.0G -> Z1:128.25G -> Z2: 77.62G -> Z3:27.0G | Z1/2通信:101.2G (2.41s) | 扁平Z3:151.9G (3.62s)
# HSDP (8机内切×2跨机同) | 常驻静态:54.0G | 机内NV:141.8G + 跨机IB仅:6.75G | 裸通信压至:0.55s
