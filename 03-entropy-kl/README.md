# 03-entropy-kl

[ZTL-03｜数学｜熵、交叉熵、困惑度与 KL](../book/03-entropy-kl.md) 正文里的可运行代码，按出现顺序编号。`exNN.out.txt` 是正文里贴出的真实输出，`python3 tools/verify.py 03-entropy-kl` 会逐字对比。

| 文件 | 位置 | 内容 |
|---|---|---|
| `ex01.py` | 第 2 章 | 在 BF16 半精度下复现显式 Softmax+log 下溢崩溃与 F.cross_entropy 平移 LogSumExp 安全输出 |
| `ex02.py` | 第 3 章 | 跨分词器 10 MB 语料 BPB 对账与防块首失明的重叠滑窗标签掩码构造脚本 |
| `ex03.py` | 第 4 章 | 验证双峰分布前反向 KL 性格反转、PyTorch F.kl_div 参数契约与 Schulman k3 估计器低方差特性 |
