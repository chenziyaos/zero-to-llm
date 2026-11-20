# 20-flash-attention

[ZTL-20｜系统｜减少注意力的内存搬运](../book/20-flash-attention.md) 正文里的可运行代码，按出现顺序编号。`exNN.out.txt` 是正文里贴出的真实输出，`python3 tools/verify.py 20-flash-attention` 会逐字对比。

| 文件 | 位置 | 内容 |
|---|---|---|
| `ex01.py` | 第 2 章 | 35 行自包含分块因果 FlashAttention：验证在线三统计量递推零误差（max_diff < 1e-6）与因果整块跳过 |
| `ex02.py` | 第 3 章 | 案例模型 全局层（24Q/4KV×256）Split-K=16 局部三元组结合律归约零误差验证（16 → 256 线程块） |
| `ex03.py` | 第 4 章 | FlashMLA 矩阵吸收恒等核验与算术强度对账：零误差免除 71 倍显存展开，解码强度从 6.0 跃至 241.8 FLOPs/B |
