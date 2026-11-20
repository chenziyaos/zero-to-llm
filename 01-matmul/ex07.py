# ZTL-01｜数学｜符号、向量与矩阵乘法 · 四个模型的每词元 KV 与 32K 上下文显存
BF16 = 2  # bytes per element

def kv_gqa(layers, kv_heads, head_dim):           # K and V for every cached layer
    return layers * 2 * kv_heads * head_dim * BF16

def kv_mla(layers, latent, rope):                  # one compressed vector per layer
    return layers * (latent + rope) * BF16

models = {
    "Qwen3-32B":   kv_gqa(64, 8, 128),
    "Qwen3.8-27B": kv_gqa(16, 4, 256),             # only the 16 global layers cache KV
    "DeepSeek-V3": kv_mla(61, 512, 64),
    "Kimi K3":     kv_mla(24, 512, 64),               # only the 24 MLA layers cache KV
}
for name, b in models.items():
    print(f"{name:12s} {b:7d} B/token = {b / 1024:6.1f} KiB/token")

ctx = 32768
for name, b in models.items():
    print(f"{name:12s} {ctx} tokens -> {b * ctx / 2**30:5.2f} GiB per sequence")

assert models["Qwen3-32B"] == 4 * models["Qwen3.8-27B"]
print("ridge H100 =", round(989 / 3.35), " ridge RTX 5090 =", round(209 / 1.79))
