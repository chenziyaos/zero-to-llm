# ZTL-07｜模型｜亲手实现分词 · 35 行零依赖完整 Byte-level BPE：覆盖训练、全局最小合并秩编码与半截 UTF-8 流式缓冲解码
from collections import Counter

def get_pairs(ids):
    return Counter(zip(ids, ids[1:]))

def merge_seq(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out

def train_bpe(text, vocab_size=268):
    ids = list(text.encode("utf-8"))
    merges, vocab = {}, {i: bytes([i]) for i in range(256)}
    for new_id in range(256, vocab_size):
        counts = get_pairs(ids)
        if not counts: break
        pair = max(counts, key=counts.get)
        ids = merge_seq(ids, pair, new_id)
        merges[pair] = new_id  # 递增的 new_id 天然充当全局唯一 Merge Rank
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
    return merges, vocab

def encode_min_rank(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        pair = min(get_pairs(ids), key=lambda p: merges.get(p, float("inf")))
        if pair not in merges: break
        ids = merge_seq(ids, pair, merges[pair])
    return ids

def stream_decode(ids, vocab):
    buf, emitted = b"", []
    for tid in ids:
        buf += vocab[tid]
        try:
            emitted.append(buf.decode("utf-8")); buf = b""
        except UnicodeDecodeError as e:
            emitted.append(buf[:e.start].decode("utf-8")); buf = buf[e.start:]
    return emitted

corpus = "aaabdaaabac    if addr == 0x10: # 中文 🚀\n    if addr == 0x10: # 中文 🚀\n"
merges, vocab = train_bpe(corpus, vocab_size=268)
enc = encode_min_rank(corpus, merges)
vocab[900], vocab[901] = b"\xe4\xb8", b"\xad"  # 模拟三字节“中”(\xe4\xb8\xad)跨两步吐出
bad = [vocab[t].decode("utf-8", errors="replace") for t in (900, 901)]
good = stream_decode([900, 901], vocab)
assert "".join(stream_decode(enc, vocab)) == corpus and good == ["", "中"]
print(f"编码词数:{len(enc)}")
print(f"裸逐词解码:{bad}")
print(f"缓冲流式:{good}")
