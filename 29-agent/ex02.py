# ZTL-29｜工程｜Agent 的循环、记忆、权限与成本 · 64 词元块级联前缀哈希模拟器：对比 pop(1) 滑窗与外部化分块折叠的 10 轮账单
P_MISS, P_HIT, P_OUT, BLOCK = 2.85, 0.285, 11.10, 64  # 美元 / 百万词元

def prefix_hashes(blocks):
    flat = [f"{tag}#{b}" for tag, tok in blocks for b in range(tok // BLOCK)]
    out, acc = [], ""
    for item in flat:
        acc = f"{acc}|{item}"
        out.append(acc)
    return out

def simulate_10_turns(mode, turns=10):
    s0 = 3840  # 静态系统指令 + MCP 工具契约
    cache = set() if mode == "1_v0_动态头+pop1" else set(prefix_hashes([("sys", s0)]))
    tot_in = hit_tok = out_tok = 0
    msgs = []
    for t in range(1, turns + 1):
        head = (f"sys_ts_{t}", s0) if mode == "1_v0_动态头+pop1" else ("sys", s0)
        bhs = prefix_hashes([head] + msgs)
        hit_b = 0
        for bh in bhs:
            if bh in cache:
                hit_b += 1
            else:
                break
        cache.update(bhs)
        tot_in += len(bhs) * BLOCK
        hit_tok += hit_b * BLOCK
        out_tok += 960
        if mode in ("1_v0_动态头+pop1", "2_静态头+pop1滑窗"):
            msgs.append((f"raw_{t}", 15360))
            if len(msgs) > 4:
                msgs.pop(0)  # 等价于完整列表执行 messages.pop(1)，后续词元位置全左移
        elif mode == "3_只追加15K原日志":
            msgs.append((f"raw_{t}", 15360))
        elif mode == "4_v2_外部化+分块折叠":
            msgs.append((f"ptr_{t}", 640))  # 200 词落盘指针摘要 + 活跃推理动作
            if s0 + sum(x[1] for x in msgs) > 36000:
                msgs = [(f"ckpt_{t}", 512)] + msgs[6:]
    miss_tok = tot_in - hit_tok
    usd = (miss_tok * P_MISS + hit_tok * P_HIT + out_tok * P_OUT) / 1e6
    print(f"{mode:20s} | 累计入:{tot_in/1e3:5.1f}K | 冷重算:{miss_tok/1e3:5.1f}K | 命中:{100*hit_tok/tot_in:5.1f}% | 成本:${usd:.3f}")

for m in ["1_v0_动态头+pop1", "2_静态头+pop1滑窗", "3_只追加15K原日志", "4_v2_外部化+分块折叠"]:
    simulate_10_turns(m)
