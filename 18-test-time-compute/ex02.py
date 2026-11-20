# ZTL-18｜后训练｜多想一会儿是否值得 · 日均 100 万混合流量四套思考预算治理方案 ROI 与双架构峰值显存对账模拟器
N_REQ = 1_000_000
P_IN, P_OUT = 0.50, 2.00         # 美元 / 百万词（1:4 输入输出价差）
TTFT_S, TPOT_S = 0.14, 0.008     # Prefill 0.14s, 单步 Decode 8.0ms
KV_QW38_KIB, GDN_MIB = 64.0, 144.0   # 案例模型: 64 KiB/词 + 144 MiB FP32 GDN定长态
KV_QW32_KIB = 256.0                  # 对照 Qwen3-32B 纯 GQA: 256 KiB/词

# 混合流量三层分布: (层级名, 占比, 平均输入词, 直答词数, 推理结论词数)
TRAFFIC = [
    ("L1-简单问答抽取", 0.65,  800, 180, 120),
    ("L2-常规分析归因", 0.25, 1100, 360, 220),
    ("L3-硬核代码数理", 0.10, 1200, 500, 260),
]

# 方案参数: ([(均思词, P95思词, 准确率%, 等效路数k), ...], 模式, 全局P95思词, 路由日费$, 路由耗时s)
# 口径铁律: 方案 B 中未闭合 </think> 的截断空答样本一律留在总分母中记错！
PLANS = {
    "V0: 全量禁思考 (enable=False)":  ([(0, 0, 94.2, 1.0), (0, 0, 72.4, 1.0), (0, 0, 51.5, 1.0)], "v0",    0,  0.0, 0.000),
    "A: 故障态全开 xhigh (上限16K)":  ([(1420, 3600, 94.0, 1.0), (2320, 5400, 84.2, 1.0), (5460, 15200, 83.4, 1.0)], "th", 5245,  0.0, 0.000),
    "B: 全量粗暴硬砍 (max_tokens=1K)":([(480, 880, 94.1, 1.0), (780, 780, 74.8, 1.0), (740, 740, 52.0, 1.0)], "th",  850,  0.0, 0.000),
    "C: 弹性分档路由 + 早停 (投产)":  ([(0, 0, 94.2, 1.0), (888, 1920, 84.2, 1.0), (1955, 5200, 88.1, 1.4)], "mix", 1095, 25.0, 0.004),
}

in_m = sum(N_REQ * w * nin for _, w, nin, _, _ in TRAFFIC) / 1e6
a_bill, a_acc = 0.0, 0.0
for name, (tiers, mode, p95_th, r_usd, r_lat) in PLANS.items():
    acc = sum(TRAFFIC[i][1] * tiers[i][2] for i in range(3))
    out_m, pk_toks = 0.0, 0
    for i, (_, w, nin, v0a, tha) in enumerate(TRAFFIC):
        na = v0a if (mode == "v0" or (mode == "mix" and i == 0)) else tha
        th_avg, th_p95, _, k_eff = tiers[i]
        out_m += N_REQ * w * k_eff * (th_avg + na) / 1e6
        pk_toks = max(pk_toks, int(nin + k_eff * (th_p95 + na)))
    qw38_mib = GDN_MIB + pk_toks * KV_QW38_KIB / 1024.0
    qw32_mib = pk_toks * KV_QW32_KIB / 1024.0
    p95_first_s = r_lat + TTFT_S + p95_th * TPOT_S
    usd = in_m * P_IN + out_m * P_OUT + r_usd
    if name.startswith("A:"):
        a_bill, a_acc = usd, acc
    print(f"{name:32s} | 综合:{acc:5.2f}% (L3:{tiers[2][2]:.1f}%) | "
          f"P95首字:{p95_first_s:4.1f}s | Qwen3.8:{qw38_mib:6.1f}MiB (32B:{qw32_mib:6.1f}MiB) | 日费:${usd:6.1f}")
print(f"投产态 C 相对故障态 A: 综合准确率 {acc - a_acc:+.2f}% | 单日账单净省 {100*(1 - usd/a_bill):.1f}%")
