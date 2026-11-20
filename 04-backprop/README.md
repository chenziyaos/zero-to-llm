# 04-backprop

[ZTL-04｜数学｜链式法则与 softmax-交叉熵的梯度](../book/04-backprop.md) 正文里的可运行代码，按出现顺序编号。`exNN.out.txt` 是正文里贴出的真实输出，`python3 tools/verify.py 04-backprop` 会逐字对比。

| 文件 | 位置 | 内容 |
|---|---|---|
| `ex01.py` | 第 2 章 | 验算三大模型栈顶词表显存尖峰，并核验分块融合线性交叉熵（Fused Chunked Linear-CE）与全量物化的梯度等价性 |
| `ex02.py` | 第 3 章 | 手写线性层与 LoRA 反向梯度：验证 B=0 首步对称破缺，并与自动微分校核误差（< 1e-7） |
| `ex03.py` | 第 4 章 | 模拟 64/93 层纯串行深网、标准 Pre-Norm 等权残差与 超深架构 Block AttnRes 的梯度保留与抗稀释表现 |
