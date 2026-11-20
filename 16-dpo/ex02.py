# ZTL-16｜后训练｜用偏好对核对 DPO · DPO / RPO / IPO / SimPO 四目标抗冗长过冲核对与 Qwen3.8-27B 8×H100 收益核算
import math

def logsigmoid(z: float) -> float:
    return -math.log1p(math.exp(-z)) if z >= 0 else z - math.log1p(math.exp(z))

# 1. 冗长负例陷阱: 精炼正例 y_w(40词, 均词-0.650) vs 废话负例 y_l(120词, 均词-0.675)
pi_w, ref_w, lw = -26.0, -28.0, 40     # 均词更优 (-0.650 > -0.675), 总和增益 +2.0
pi_l, ref_l, ll = -81.0, -88.0, 120    # 借 120 词乘法杠杆攒出总和增益 +7.0!
m_dpo = 0.10 * ((pi_w - ref_w) - (pi_l - ref_l))       # 标准 DPO 边际 -0.500 < 0 误罚精炼正例!
l_dpo = -logsigmoid(m_dpo)
l_rpo = l_dpo + 0.10 * (-pi_w / lw)                    # RPO 叠加 0.1*NLL 护住正例绝对似然
m_sim = 2.50 * (pi_w / lw - pi_l / ll) - 0.02          # SimPO 均词打分 m=+0.0425 > 0 拨乱反正!
diff, tau = 12.0, 0.25                                 # 确定性偏好过冲 diff=12.0 远超平衡点 1/(2*tau)=2.0
g_dpo, g_ipo = 0.10 / (1.0 + math.exp(0.10 * diff)), 2.0 * (diff - 1.0 / (2.0 * tau))

# 2. 8xH100 (峰值 8*989 TFLOPS) 对齐 Qwen3.8-27B (27B, 30000对x2轮, 均长2048) 耗时与解码账
d_trn, d_ref, peak = 30000 * 2 * 2 * 2048, 30000 * 2 * 2048, 8 * 989e12
t_base = (8.0 * 27e9 * d_trn) / (peak * 0.333) / 3600
t_opt  = (2.0 * 27e9 * d_ref) / (peak * 0.550) / 3600 + (6.0 * 27e9 * d_trn) / (peak * 0.412) / 3600
print(f"[冗长与限幅] DPO_m={m_dpo:+.3f}(L={l_dpo:.4f}) | RPO_L={l_rpo:.4f} | SimPO_m={m_sim:+.4f} | IPO回拉梯度={g_ipo:+.1f}(DPO外推={g_dpo:.4f})")
print(f"[均词对数概率] 精炼正例={pi_w/lw:.3f} nats | 冗长负例={pi_l/ll:.3f} nats")
print(f"[8xH100账单] 双前向基线={t_base:.2f}h | 预存优化={t_opt:.2f}h(提速{(t_base/t_opt-1)*100:.1f}%) | 解码词缩减={(1-395/1120)*100:.1f}%")
