# ZTL-01｜数学｜符号、向量与矩阵乘法 · Qwen3-32B 上投影的屋脊线上限
GPUS = {"H100": (989, 3.35), "RTX 5090": (209, 1.79)}   # dense BF16 TFLOPS, TB/s
K, N = 5120, 25600                                       # Qwen3-32B up_proj

def gemm(M, K, N, bytes_per=2):
    flops = 2 * M * N * K
    moved = bytes_per * (M * K + K * N + M * N)          # read x, read W, write y once
    return flops, moved, flops / moved

for name, (tf, bw) in GPUS.items():
    print(f"{name:8s} ridge = {tf / bw:.0f} FLOP/byte")
for M in (1, 64, 128, 256, 4096):
    f, b, ai = gemm(M, K, N)
    h100 = min(989, ai * 3.35)                           # roofline bound, TFLOPS
    rtx = min(209, ai * 1.79)
    print(f"M={M:5d}  {f / 1e9:8.2f} GFLOP  {b / 1e6:6.1f} MB  AI {ai:7.1f}  "
          f"H100 {h100:5.1f} TF ({f / h100 / 1e9:.3f} ms)  RTX 5090 {rtx:5.1f} TF")

for name, (tf, bw) in GPUS.items():
    M = 1
    while gemm(M, K, N)[2] < tf / bw:
        M += 1
    print(f"{name:8s} becomes compute-bound at M = {M}")

E, top, B = 256, 8, 64                                   # DeepSeek-V3 routed experts
hit = 1 - (1 - top / E) ** B                             # chance an expert gets >=1 token
print(f"DeepSeek-V3 B={B}: {hit:.0%} of experts touched, "
      f"{B * top / (E * hit):.1f} tokens per touched expert")
