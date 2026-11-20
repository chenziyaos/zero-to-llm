# 21-quantization

[ZTL-21｜系统｜用精度换内存要付什么代价](../book/21-quantization.md) 正文里的可运行代码，按出现顺序编号。`exNN.out.txt` 是正文里贴出的真实输出，`python3 tools/verify.py 21-quantization` 会逐字对比。

| 文件 | 位置 | 内容 |
|---|---|---|
| `ex01.py` | 第 1 章 | Per-Tensor 单刻度坍缩 vs Group-128 分组量化误差及 Qwen3.8-27B 真实显存拆账 |
| `ex02.py` | 第 3 章 | SmoothQuant 对角迁移降噪 W8A8 与 AWQ 1% 关键通道放大护航 W4A16 实证 |
| `ex03.py` | 第 5 章 | Qwen3.8-27B 在 RTX 4090 (AWQ) / RTX 5090 (FP8) / H100 (全栈 FP8) 的端云总账核算器 |
