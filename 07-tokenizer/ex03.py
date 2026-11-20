# ZTL-07｜模型｜亲手实现分词 · 四大前沿架构词表汇率对账与单卡 H100 静态嵌入换动态 KV 盈亏平衡推演脚本
import math

# (模型名, 词表V, 隐藏维d, 总参B, 汇率r, 每词线性KV字节, 定长态MiB)
# r = 每词平均字节数，是本篇取的示意值，不是实测；真实值随语料和语言比例变化
MODELS = [
    ("DeepSeek-V3",  129280, 7168,  671.0, 3.55, 61 * 576 * 2,    0.0),
    ("Qwen3-32B",    151936, 5120,   32.8, 3.60, 64 * 2048 * 2,   0.0),
    ("Kimi-K3",      163840, 7168, 2800.0, 3.82, 24 * 576 * 2,  207.0),
    ("Qwen3.8-27B",  248320, 5120,   26.9, 4.32, 16 * 2048 * 2, 144.0),
]
REPO_BYTES = 100_000  # 100 KB 中英代码混合源码输入

for name, v, d, tot_b, r, kv_b, _ in MODELS:
    p_voc_b = 2 * v * d / 1e9
    bf16_gb = p_voc_b * 2
    s_tok = round(REPO_BYTES / r)
    kv_gb = s_tok * kv_b / 1e9
    print(f"{name:11s} | V={v:6d} | 2Vd={p_voc_b:.2f}B ({100*p_voc_b/tot_b:4.1f}%) | 词表BF16={bf16_gb:.2f}GB | S={s_tok:5d} | 线性KV={kv_gb:.3f}GB")

# 核算 Qwen3.8-27B 词表从 151,936 扩至 248,320 的单卡 H100 回本并发数
extra_vocab_gb = 2 * (248320 - 151936) * 5120 * 2 / 1e9
s_151k, s_248k = round(REPO_BYTES / 3.60), round(REPO_BYTES / 4.32)
saved_kv_req_gb = (s_151k - s_248k) * 65536 / 1e9
be_batch = math.ceil(extra_vocab_gb / saved_kv_req_gb)
assert round(extra_vocab_gb, 2) == 1.97 and be_batch == 7
print(f"扩表多付静态权重:+{extra_vocab_gb:.2f}GB | 100KB省词:{s_151k-s_248k} (-16.7%) | 单路省KV:{saved_kv_req_gb:.3f}GB | 回本并发:B>={be_batch}")
