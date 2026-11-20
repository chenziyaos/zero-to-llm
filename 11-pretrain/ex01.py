# ZTL-11｜训练｜从语料到一轮小预训练 · 对齐 Qwen3.8-27B（248,320 词表）的 uint32 memmap 打包、cu_seqlens 与线性状态 reset_state 双隔离验证
import tempfile, numpy as np

VOCAB_SIZE = 248320      # Qwen3.8-27B 真实词表，远超 uint16 上限 65535
EOS_ID = 248044          # 高位特殊结束符 <|endoftext|>，必须用 uint32 存储
assert VOCAB_SIZE > 65535, "大词表严禁使用 uint16，否则高位词号发生模 65536 溢出！"

docs = [[198001, 205112, 8812], [241009, 7731], [130500, 130501, 130502]]
stream = np.array([tok for d in docs for tok in (d + [EOS_ID])], dtype=np.uint32)

with tempfile.NamedTemporaryFile(suffix=".bin") as f:
    stream.tofile(f.name)
    mmap_arr = np.memmap(f.name, dtype=np.uint32, mode="r")
    S = 9
    buf = mmap_arr[: S + 1].astype(np.int64)
    x, y = buf[:-1], buf[1:].copy()
    y[x == EOS_ID] = -100  # 跨篇边界屏蔽下一篇首词预测损失

    # 推导新文档起点掩码 reset_mask、文档号 doc_ids、篇内归零 pos_ids 与 cu_seqlens
    reset_mask = np.zeros(S, dtype=bool)
    reset_mask[0] = True
    reset_mask[1:] = (x[:-1] == EOS_ID)
    doc_ids = np.cumsum(reset_mask) - 1
    idx = np.arange(S)
    pos_ids = idx - np.maximum.accumulate(reset_mask.astype(np.int64) * idx)
    boundaries = np.where(reset_mask)[0].tolist() + [S]
    cu_seqlens = np.array(boundaries, dtype=np.int32)

    # 验证 GDN / KDA 线性递推状态 S_t 与 kernel=4 因果短卷积边界清零隔离
    rng = np.random.RandomState(0)
    k_seq, v_seq = rng.randn(S, 4), rng.randn(S, 4)

    def run_linear_layer(k_in, v_in, r_mask):
        state = np.zeros((4, 4), dtype=np.float32)
        conv_buf = np.zeros((3, 4), dtype=np.float32)  # conv_kernel=4 历史抽头
        outs = []
        for t in range(len(k_in)):
            if r_mask[t]:
                state.fill(0.0)     # 铁律 1：跨过 EOS 清零递推状态矩阵 S_t
                conv_buf.fill(0.0)  # 铁律 2：同步清空因果 conv1d 滑窗缓存
            kt = 0.5 * k_in[t] + 0.5 * conv_buf.mean(axis=0)
            conv_buf = np.vstack([conv_buf[1:], k_in[t]])
            state = 0.9 * state + np.outer(kt, v_in[t])
            outs.append(state.T @ kt)
        return np.stack(outs)

    out_base = run_linear_layer(k_seq, v_seq, reset_mask)
    k_dirty = k_seq.copy(); k_dirty[:4] += 999.0  # 剧烈篡改第 0 篇内容模拟前文噪声
    out_dirty = run_linear_layer(k_dirty, v_seq, reset_mask)
    leak_diff = float(np.max(np.abs(out_base[4:] - out_dirty[4:])))
    print("x (输入词号) :", x.tolist())
    print("y (标签，-100表示屏蔽) :", y.tolist())
    print("cu_seqlens :", cu_seqlens.tolist(), "| pos_ids :", pos_ids.tolist())
    print("reset_mask :", reset_mask.astype(int).tolist(), f"| 后篇线性层跨文档最大泄露误差: {leak_diff:.1f}")
