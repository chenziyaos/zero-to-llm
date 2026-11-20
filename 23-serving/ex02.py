# ZTL-23｜系统｜把模型部署成服务 · 四代架构 16K 跨机 RDMA 传输耗时、xPyD 配比及含潮汐因子百万词元真实成本核算器
import math

MODELS = [
    ("Legacy-MHA (2MiB/tok)",     2097152,   0.0, 100.0),
    ("Qwen3.8-27B (64KiB+GDN)",     65536, 151.0, 400.0),
    ("DeepSeek-V3 (FP8 MLA)",       35136,   0.0, 400.0),
]
for name, b_kv, state_mb, nic_gbps in MODELS:
    L = 16000
    payload_mb = (L * b_kv) / 1e6 + state_mb
    bw_mbs = (nic_gbps / 8.0) * 0.90 * 1e3          # 90% RDMA 有效载荷带宽 (MB/s)
    print(f"{name:26s} | 载荷:{payload_mb:8.2f} MB | {nic_gbps:3.0f}G耗时:{payload_mb/bw_mbs*1e3:7.2f} ms")

def calc_cost_1m(c_gpu_hr: float, n_p: int, n_d: int, goodput_req_s: float, tok_req: int, eta: float, fx: float = 7.20):
    usd = (c_gpu_hr * (n_p + n_d)) / (goodput_req_s * tok_req * 3600.0 * eta) * 1e6
    return round(usd, 3), round(usd * fx, 2)

print("V0 改造前 (24卡混部, 10.8r/s):", calc_cost_1m(2.40, 24, 0, 10.8, 16500, 0.55))
print("V3 改造后 (3P5D共8卡,11.8r/s):", calc_cost_1m(2.40, 3, 5, 11.8, 16500, 0.55))
