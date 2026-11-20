# ZTL-15｜后训练｜只训练一小部分参数 · 案例模型 在 RTX 4090 (24GiB) / 5090 (32GiB) / H100 (80GiB) 单卡微调峰值显存对账脚本
GIB = 1024 ** 3 / 1e9
L, L_GDN, L_GQA, D, D_FF, V = 64, 48, 16, 5120, 17408, 248320
PSI_NOM = 27.0e9
PSI_LORA = (L * 3 * (D + D_FF) + L_GQA * (29696 + 11264) + L_GDN * (21504 + 11264)) * 16  # 104,857,600

def audit_peak(name: str, qlora: bool, seq_len: int, ckpt: bool, chunk_ce: bool, cap_gib: float):
    m_static = (PSI_NOM * 4.127 / 8 + 0.43e9 + 8 * PSI_LORA) / 1e9 if qlora else (2 * PSI_NOM + 16 * PSI_LORA) / 1e9
    b_layer = 2 * seq_len * (2 * D + 3 * D_FF + 29696) / 1e9   # 单层未重算前向激活 (184.3 KB/词)
    m_saved = (L * 2 * seq_len * D / 1e9) if ckpt else (L * b_layer)
    m_recomp = b_layer if ckpt else 0.0                        # 反向逐层重算单层瞬态缓冲
    m_deq_head = (2 * V * D / 1e9) if qlora else 0.0           # QLoRA 算输出头时 lm_head 反量化缓冲 (2.54G)
    m_head = m_deq_head + ((256 * V * 4 / 1e9) if chunk_ce else (6 * seq_len * V / 1e9))
    peak_gb = m_static + m_saved + max(m_head, m_recomp) + 0.14
    peak_gib = peak_gb / GIB
    tag = "PASS" if peak_gib <= cap_gib - 2.0 else "OOM"
    print(f"{name:27s} | 静态:{m_static:5.2f}G | 激活:{m_saved:6.2f}G | 峰值:{peak_gb:6.2f}G ({peak_gib:5.1f}/{cap_gib:.0f}GiB) [{tag}]")

audit_peak("4090 QLoRA 4K (关检查点)",    True,  4096,  False, False, 24.0)
audit_peak("4090 QLoRA 4K (仅开检查点)",  True,  4096,  True,  False, 24.0)
audit_peak("4090 QLoRA 4K (+分块融合CE)", True,  4096,  True,  True,  24.0)
audit_peak("5090 QLoRA 16K (+分块融合CE)",True,  16384, True,  True,  32.0)
audit_peak("H100 LoRA 16K (关检查点)",    False, 16384, False, False, 80.0)
audit_peak("H100 LoRA 16K (仅开检查点)",  False, 16384, True,  False, 80.0)
audit_peak("H100 LoRA 16K (+分块融合CE)", False, 16384, True,  True,  80.0)
