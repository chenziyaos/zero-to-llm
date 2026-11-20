# ZTL-24｜系统｜长上下文与混合专家怎样改变账单 · 200K 代码仓连续 8 轮追问场景下商业缓存 API 与两档自建集群的盈亏平衡模拟器
import math

GPU_USD_DAY = 2.50 * 24.0   # H100 全包折旧含电费: $2.50/卡时 = $60/卡天
REPO_TOK, TURNS, Q_TOK, OUT_TOK = 200_000, 8, 1_000, 500

def session_tokens(repo_hit=0.80, hist_hit=0.98):
    miss, hit = REPO_TOK * (1.0 - repo_hit) + Q_TOK, REPO_TOK * repo_hit
    for r in range(1, TURNS):
        hist = REPO_TOK + r * (Q_TOK + OUT_TOK)
        miss += hist * (1.0 - hist_hit) + Q_TOK
        hit  += hist * hist_hit
    return miss / 1e6, hit / 1e6, (TURNS * OUT_TOK) / 1e6

miss_m, hit_m, out_m = session_tokens()
api_sess = miss_m * 0.30 + hit_m * 0.03 + out_m * 1.20  # 缓存API: $0.30/$0.03/$1.20 每M
no_cache = (miss_m + hit_m) * 0.30 + out_m * 1.20       # 破坏前缀全Miss时的冤枉钱对照
print(f"单次8轮会话: Miss={miss_m*1e3:.1f}K, Hit={hit_m*1e3:.1f}K(占95.3%), Out={out_m*1e3:.1f}K")
print(f"API单会话成本: 开启缓存=${api_sess:.4f} vs 头部塞动态时间戳全Miss=${no_cache:.4f} (贵{no_cache/api_sess:.1f}倍)")

# 自建实例池规格: (方案名, 每组H100卡数, 扣除潮汐eta=0.55后单组日承载8轮会话上限)
pools = [("2xH100 Qwen3.8-27B", 2, 2800), ("32xH100 万亿MoE", 32, 36000)]
print(f"盈亏平衡点: 2xH100跨过 {120/api_sess:.0f} 会话/天; 32xH100跨过 {1920/api_sess:.0f} 会话/天")
for sess_d in [500, 2500, 12500, 35000, 100000]:
    turns_k = sess_d * TURNS // 1000
    nq = math.ceil(sess_d / pools[0][2])
    cq = (nq * pools[0][1] * GPU_USD_DAY) / sess_d
    nm = math.ceil(sess_d / pools[1][2])
    cm = (nm * pools[1][1] * GPU_USD_DAY) / sess_d
    print(f"日追问={turns_k:4d}K次({sess_d:6d}会话) | API=${api_sess:.3f} | 2xH100(x{nq:2d})=${cq:.3f} | 32xH100(x{nm})=${cm:.3f}")
