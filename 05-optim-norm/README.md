# 05-optim-norm

[ZTL-05｜数学｜优化、归一化与数值稳定](../book/05-optim-norm.md) 正文里的可运行代码，按出现顺序编号。`exNN.out.txt` 是正文里贴出的真实输出，`python3 tools/verify.py 05-optim-norm` 会逐字对比。

| 文件 | 位置 | 内容 |
|---|---|---|
| `ex01.py` | 第 1 章 | 64 层残差流的尺度增长模拟 |
| `ex02.py` | 第 2 章 | 手写全局 L2 保向裁剪、解耦 AdamW（对齐 PyTorch）与 MuonClip 的 QK-Clip 权重钳制验证 |
| `ex03.py` | 第 3 章 | 精确复现纯 BF16 下 1e-4 微步更新 1000 步原地踏步，并验证 FP32 主权重与 1x128 分块 FP8 修复效果 |
| `ex04.py` | 第 4 章 | 验证三统计量在线 Softmax 在极端大分数（1000+）下单遍流式递推与标准路径零误差等价 |
