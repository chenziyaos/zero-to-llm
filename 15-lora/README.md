# 15-lora

[ZTL-15｜后训练｜只训练一小部分参数](../book/15-lora.md) 正文里的可运行代码，按出现顺序编号。`exNN.out.txt` 是正文里贴出的真实输出，`python3 tools/verify.py 15-lora` 会逐字对比。

| 文件 | 位置 | 内容 |
|---|---|---|
| `ex01.py` | 第 2 章 | 自包含 PyTorch LoRALinear 模块：断言核验第 0 步逐比特恒等、首步梯度流向与 FP32 离线合并等价 |
| `ex02.py` | 第 3 章 | 案例模型 在 RTX 4090 (24GiB) / 5090 (32GiB) / H100 (80GiB) 单卡微调峰值显存对账脚本 |
