# 12-multi-gpu

[ZTL-12｜训练｜多卡的账和中断后续上](../book/12-multi-gpu.md) 正文里的可运行代码，按出现顺序编号。`exNN.out.txt` 是正文里贴出的真实输出，`python3 tools/verify.py 12-multi-gpu` 会逐字对比。

| 文件 | 位置 | 内容 |
|---|---|---|
| `ex01.py` | 第 2 章 | Qwen3.8-27B（Ψ=27B）在 8 卡与 16 卡 H100 下的 ZeRO 阶梯及 HSDP 跨机通信核算 |
| `ex02.py` | 第 4 章 | PyTorch DCP 三阶段原子提交范式与第 10 步断电续跑至第 20 步逐比特零误差验证 |
