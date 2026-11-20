# ZTL-03｜数学｜熵、交叉熵、困惑度与 KL · 跨分词器 10 MB 语料 BPB 对账与防块首失明的重叠滑窗标签掩码构造脚本
import math

def build_strided_eval_windows(token_ids: list[int], window: int, stride: int) -> list[tuple[list[int], list[int]]]:
    """构造无偏重叠滑窗：返回 (input_ids, target_labels)，前缀与重复位严格置 -100"""
    n, pairs, prev_end = len(token_ids), [], 0
    for begin in range(0, n, stride):
        end = min(begin + window, n)
        trg_len = end - prev_end  # 本窗真正需要计费的新增右侧词元数
        inp = token_ids[begin:end]
        # 自回归左移对齐：位置 i 预测 inp[i+1]；仅保留末尾 trg_len 个新增预测目标
        raw_targets = inp[1:]
        keep = trg_len - 1 if begin == 0 else trg_len
        masked_targets = [-100] * (len(raw_targets) - keep) + raw_targets[-keep:]
        pairs.append((inp[:-1], masked_targets))
        prev_end = end
        if end == n:
            break
    return pairs

# 1. 验证长度为 10 的玩具序列在 window=6, stride=3 下每个非首词恰好计费 1 次（共 9 个有效目标）
wins = build_strided_eval_windows(list(range(10)), window=6, stride=3)
valid_targets = [t for _, lbl in wins for t in lbl if t != -100]
assert valid_targets == list(range(1, 10)), f"计费目标漏重异常: {valid_targets}"
print(f"无偏滑窗验证: {len(valid_targets)} 个目标恰好各计费 1 次")

# 2. 核算 10 MB (10,000,000 UTF-8 字节) 语料上 32K 基座与 Qwen3.8-27B (248K) 的 PPL/BPB 翻转
N_BYTES = 10_000_000
for name, tok_cnt, mean_nll in [("32K 旧基座", 3_300_000, 1.9879), ("Qwen3.8-27B", 1_940_000, 2.7973)]:
    total_nll = tok_cnt * mean_nll
    ppl, bpb = math.exp(mean_nll), total_nll / (N_BYTES * math.log(2.0))
    print(f"{name:12s} -> 表面 PPL={ppl:5.2f} | 全文总比特={total_nll/math.log(2)/1e6:.3f} Mbits | 公平 BPB={bpb:.4f}")
