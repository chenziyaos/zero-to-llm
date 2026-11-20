# ZTL-14｜后训练｜只对该学的回答算损失 · 构造多轮工具排障轨迹无泄漏 labels 并断言背景与工具观察梯度严格归零
import numpy as np

# 对齐 Qwen3.8-27B 特权控制符专号 (vocab_size=248320, pad=endoftext=248044, im_start=248045, im_end=eos=248046)
TOK = {
    "<|endoftext|>": 248044, "<|im_start|>": 248045, "<|im_end|>": 248046, "\n": 198,
    "system": 8948, "user": 872, "assistant": 77091, "tool": 14880,
    "SRE_SPEC": 3101, "ERR_504_LOG": 4205, "<think>": 248047, "check_pool": 5309,
    "</think>": 248048, "<tool_call>query_pod</tool_call>": 6412,
    '{"db_wait_ms":4790}': 7520, "pool_full": 8631, "扩连接池至128。": 9742,
}

def encode_message(role: str, content_tokens: list[str]) -> tuple[list[int], list[int]]:
    header = [TOK["<|im_start|>"], TOK[role], TOK["\n"]]
    body = [TOK[t] for t in content_tokens]
    footer = [TOK["<|im_end|>"], TOK["\n"]]
    ids = header + body + footer
    if role == "assistant":
        # 引导头 3 词与回合尾换行填 -100；仅助手生成正文（含思考与工具调用）及 <|im_end|> 留真实 ID
        lbs = [-100, -100, -100] + body + [TOK["<|im_end|>"], -100]
    else:
        lbs = [-100] * len(ids)
    return ids, lbs

tau_0 = [
    ("system", ["SRE_SPEC"]),
    ("user", ["ERR_504_LOG"]),
    ("assistant", ["<think>", "check_pool", "</think>", "<tool_call>query_pod</tool_call>"]),
    ("tool", ['{"db_wait_ms":4790}']),
    ("assistant", ["<think>", "pool_full", "</think>", "扩连接池至128。"]),
]
input_ids, labels = [], []
for role, seg in tau_0:
    i_seg, l_seg = encode_message(role, seg)
    input_ids.extend(i_seg); labels.extend(l_seg)

# 执行因果右移切片并按融合交叉熵雅可比公式 (p - y) / ∑m_t 核验逐位置反向梯度
id_map = {v: idx for idx, v in enumerate(TOK.values())}
shift_y = np.array([id_map[l] if l != -100 else -100 for l in labels[1:]])
probs = np.full((len(input_ids) - 1, len(TOK)), 1.0 / len(TOK), dtype=np.float64)
valid = (shift_y != -100)
grad_logits = np.zeros_like(probs)
grad_logits[valid] = probs[valid]
grad_logits[np.where(valid)[0], shift_y[valid]] -= 1.0
grad_logits /= int(valid.sum())

sup_pos = np.where(valid)[0].tolist()
masked_max_grad = float(np.linalg.norm(grad_logits[~valid], axis=-1).max())
sup_start_inputs = [input_ids[sup_pos[0]], input_ids[sup_pos[5]]]
sup_end_targets = [labels[sup_pos[4] + 1], labels[sup_pos[-1] + 1]]

print(f"总词元 T={len(input_ids)} | 预测对={len(shift_y)} | 有效监督词 ∑m_t={len(sup_pos)}")
print(f"激活监督起跑下标: {[sup_pos[0], sup_pos[5]]} -> 对应输入词 ID: {sup_start_inputs} (均为换行 198)")
print(f"两轮收尾目标 ID: {sup_end_targets} (均为 <|im_end|> 248046) | 屏蔽区最大梯度: {masked_max_grad:.1e}")
assert masked_max_grad == 0.0 and sup_end_targets == [248046, 248046]
