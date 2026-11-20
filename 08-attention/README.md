# 08-attention

[ZTL-08｜模型｜把注意力写成矩阵](../book/08-attention.md) 正文里的可运行代码，按出现顺序编号。`exNN.out.txt` 是正文里贴出的真实输出，`python3 tools/verify.py 08-attention` 会逐字对比。

| 文件 | 位置 | 内容 |
|---|---|---|
| `ex01.py` | 第 2 章 | 自包含 PyTorch Gated-GQA 模块（含独立头宽 d_k=256、QK-Norm、输出门控与 Prefill/Decode 误差 < 1e-5 单测） |
| `ex02.py` | 第 3 章 | 25 行零依赖 NumPy 核验：证明 DeepSeek MLA 显式键值展开态与解码矩阵吸收态达到 10^-16 机器精度等价 |
| `ex03.py` | 第 4 章 | 对比标准线性累加淹没旧信息 vs GDN/KDA 秩-1 Delta 规则精准擦除覆写键值对 |
| `ex04.py` | 第 5 章 | 零依赖纯 Python 总账核验：一键复算 2020→2026 六代注意力每词缓存 |
